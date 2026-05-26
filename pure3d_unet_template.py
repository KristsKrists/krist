# -*- coding: utf-8 -*-
"""
Pure 3D U-Net baseline (no Mamba blocks) for multi-class 3D segmentation.

Same dataset layout, patch tiling, losses, and merged-volume eval as mamba3d_unet_template.py.
"""

from __future__ import annotations

import os
import math
import time
import pickle
import random
import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from patchify import patchify
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# =============================================================================
# === USER CONFIG ==============================================================
# =============================================================================

DATASET_ROOT = os.environ.get("SEG_DATASET_ROOT", "./data/segmentation")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints_pure_unet")
TRAINING_MODE = os.environ.get("TRAINING_MODE", "new")  # "new" | "continue"

HEIGHT = 512
WIDTH = 512
PATCH_D = 8
PATCH_H = HEIGHT // 4
PATCH_W = WIDTH // 4

VAL_STEP = (PATCH_H // 2, PATCH_W // 2, PATCH_D)  # (64, 64, 8)
INFER_STEP = (PATCH_H // 2, PATCH_W // 2, PATCH_D // 2)  # (64, 64, 4)
TRAIN_STEP = (PATCH_H, PATCH_W, PATCH_D // 2)  # (128, 128, 4)

EPOCHS = int(os.environ.get("EPOCHS", "5000"))
PATIENCE = int(os.environ.get("PATIENCE", "20"))
_max_pat = os.environ.get("MAX_PATIENTS", "135")
MAX_PATIENTS: Optional[int] = None if _max_pat.lower() in ("", "none", "all") else int(_max_pat)

BASE_CHANNELS = int(os.environ.get("BASE_CHANNELS", "16"))

AUG_PROB = 0.5
FOREGROUND_THRESHOLD = 0.0

INITIAL_LR = 1e-5
MAX_LR = 3e-4
CLR_STEP_SIZE = 2290

WEIGHT_DECAY = 1e-5

CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "EDH",
    2: "ICH",
    3: "IVH",
    4: "SAH",
    5: "SDH",
}

CT_WINDOWS = [
    [40, 80],
    [600, 3000],
    [75, 215],
    [60, 120],
]

_STARTING_CLASS_WEIGHTS = [0.171, 198.741, 15.075, 52.645, 58.868, 28.757]


def dampened_class_weights(starting: List[float]) -> List[float]:
    min_w = min(starting)
    normalized = [w / min_w for w in starting]
    dampened = [math.sqrt(w) for w in normalized]
    return [round(w, 3) for w in dampened]


CLASS_WEIGHT_VALUES = dampened_class_weights(_STARTING_CLASS_WEIGHTS)


# =============================================================================
# Utilities (preprocess, augmentation, tiling)
# =============================================================================

def apply_window(image_hu: np.ndarray, window: List[float]) -> np.ndarray:
    center, width = float(window[0]), float(window[1])
    lower = center - 0.5 - (width - 1) / 2
    upper = center - 0.5 + (width - 1) / 2
    windowed = np.clip(image_hu, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8)
    return windowed.astype(np.float32, copy=False)


def standardize_windows(image_hu: np.ndarray, windows: List[List[float]]) -> np.ndarray:
    chans = [apply_window(image_hu, w) for w in windows]
    return np.stack(chans, axis=-1).astype(np.float32, copy=False)


def normalize_image01(image: np.ndarray) -> np.ndarray:
    mn = float(np.min(image))
    mx = float(np.max(image))
    if mx - mn <= 0:
        return image.astype(np.float32, copy=False)
    return ((image - mn) / (mx - mn)).astype(np.float32, copy=False)


def one_hot_np(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = mask.astype(np.int64, copy=False)
    return np.eye(num_classes, dtype=np.float32)[mask]


def pad_depth_axis_background(img: np.ndarray, mask_oh: np.ndarray, pad_amount: int) -> Tuple[np.ndarray, np.ndarray]:
    if pad_amount <= 0:
        return img, mask_oh
    h, w, _d, c = img.shape
    _h2, _w2, _d2, k = mask_oh.shape
    img_tail = np.zeros((h, w, pad_amount, c), dtype=np.float32)
    mask_tail = np.zeros((h, w, pad_amount, k), dtype=np.float32)
    mask_tail[..., 0] = 1.0
    return np.concatenate([img, img_tail], axis=2), np.concatenate([mask_oh, mask_tail], axis=2)


def zoom(img: np.ndarray, mask: np.ndarray, p: float = AUG_PROB) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > p:
        return img, mask

    zoom_factor = random.uniform(0.7, 1.3)
    h, w = img.shape[0], img.shape[1]
    new_h = max(1, int(round(h * zoom_factor)))
    new_w = max(1, int(round(w * zoom_factor)))

    aa = zoom_factor < 1
    img_scaled = np.stack(
        [resize(img[..., c], (new_h, new_w), order=1, preserve_range=True, anti_aliasing=aa) for c in range(img.shape[-1])],
        axis=-1,
    )
    mask_scaled = np.stack(
        [resize(mask[..., c], (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False) for c in range(mask.shape[-1])],
        axis=-1,
    )

    def _fit_spatial(vol: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        out = np.asarray(vol)
        hh, ww = out.shape[0], out.shape[1]
        if hh > target_h:
            top = (hh - target_h) // 2
            out = out[top : top + target_h, :, :, :]
            hh = target_h
        if ww > target_w:
            left = (ww - target_w) // 2
            out = out[:, left : left + target_w, :, :]
            ww = target_w
        hh, ww = out.shape[0], out.shape[1]
        if hh < target_h or ww < target_w:
            pad_top = (target_h - hh) // 2
            pad_bottom = target_h - hh - pad_top
            pad_left = (target_w - ww) // 2
            pad_right = target_w - ww - pad_left
            try:
                out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0)), mode="reflect")
            except ValueError:
                out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
        return out

    return _fit_spatial(img_scaled, h, w), _fit_spatial(mask_scaled, h, w)


def mirroring(img: np.ndarray, mask: np.ndarray, p: float = AUG_PROB) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > p:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() > p:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    return img, mask


def random_rotate(img: np.ndarray, mask: np.ndarray, p: float = AUG_PROB) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > p:
        return img, mask
    k = random.choice([1, 2, 3])
    img_out = np.empty_like(img)
    mask_out = np.empty_like(mask)
    for d in range(img.shape[2]):
        img_out[:, :, d, :] = np.rot90(img[:, :, d, :], k=k, axes=(0, 1))
        mask_out[:, :, d, :] = np.rot90(mask[:, :, d, :], k=k, axes=(0, 1))
    return img_out, mask_out


@dataclass
class VolumeBatch:
    patches_img: np.ndarray
    patches_mask: np.ndarray


class PatchGenerator:
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: Tuple[int, int, int, int],
        step_size: Tuple[int, int, int],
        max_patients: Optional[int],
        augment: bool,
        shuffle: bool,
        threshold: float,
        num_classes: int,
        windows: List[List[float]],
    ):
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii")])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii")])
        if max_patients is not None:
            self.image_files = self.image_files[: int(max_patients)]
            self.mask_files = self.mask_files[: int(max_patients)]
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Mismatched images ({len(self.image_files)}) and masks ({len(self.mask_files)})")

        self.patch_size = patch_size
        self.step_size = step_size
        self.augment = augment
        self.shuffle = shuffle
        self.threshold = float(threshold)
        self.num_classes = int(num_classes)
        self.windows = windows
        self.indices = np.arange(len(self.image_files))
        self.on_epoch_end()
        self.patch_voxels = int(np.prod(self.patch_size[:3]))

    def __len__(self) -> int:
        return len(self.image_files)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(self.image_files[idx], mmap=True).get_fdata(dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        mask = nib.load(self.mask_files[idx], mmap=True).get_fdata().astype(np.uint8, copy=False)
        return img, mask

    def _preprocess(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = normalize_image01(standardize_windows(img, self.windows))
        mask_oh = one_hot_np(mask, self.num_classes).astype(np.float32, copy=False)

        if self.augment and random.random() > AUG_PROB:
            img, mask_oh = zoom(img, mask_oh, p=AUG_PROB)
            img, mask_oh = mirroring(img, mask_oh, p=AUG_PROB)
            img, mask_oh = random_rotate(img, mask_oh, p=AUG_PROB)

        img = resize(img, (HEIGHT, WIDTH, img.shape[2], img.shape[3]), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32, copy=False)
        mask_oh = resize(mask_oh, (HEIGHT, WIDTH, mask_oh.shape[2], self.num_classes), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)
        return img, mask_oh

    def _load_and_preprocess(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._preprocess(*self._load_raw(idx))

    def _patchify_all(self, img: np.ndarray, mask_oh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        ph, pw, pd, cin = self.patch_size
        sh, sw, sd = self.step_size

        remainder = (img.shape[2] - pd) % sd
        if remainder != 0:
            img, mask_oh = pad_depth_axis_background(img, mask_oh, sd - remainder)

        patches_img = patchify(img, (ph, pw, pd, cin), step=(sh, sw, sd, cin))
        patches_mask = patchify(mask_oh, (ph, pw, pd, self.num_classes), step=(sh, sw, sd, self.num_classes))
        nh, nw, nd = patches_img.shape[0], patches_img.shape[1], patches_img.shape[2]
        patches_img = patches_img.reshape(-1, ph, pw, pd, cin).astype(np.float32, copy=False)
        patches_mask = patches_mask.reshape(-1, ph, pw, pd, self.num_classes).astype(np.float32, copy=False)
        return patches_img, patches_mask, (nh, nw, nd)

    def get_train_patches(self, patient_i: int) -> VolumeBatch:
        idx = int(self.indices[patient_i])
        img_raw, mask_raw = self._load_raw(idx)
        empty = VolumeBatch(
            patches_img=np.empty((0,) + self.patch_size, np.float32),
            patches_mask=np.empty(
                (0, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.num_classes),
                np.float32,
            ),
        )

        for _ in range(100):
            img, mask_oh = self._preprocess(img_raw, mask_raw)
            if float(mask_oh[..., 1:].sum()) <= 0.0:
                continue

            patches_img, patches_mask, _grid = self._patchify_all(img, mask_oh)
            fg_vox = patches_mask[..., 1:].sum(axis=(1, 2, 3, 4))
            keep = (fg_vox / max(1.0, float(self.patch_voxels))) > self.threshold
            if np.any(keep):
                return VolumeBatch(patches_img=patches_img[keep], patches_mask=patches_mask[keep])
            return empty

        raise RuntimeError("Unable to generate a non-empty patch batch after multiple retries.")

    def get_volume_for_merge(self, patient_i: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]]:
        img, mask_oh = self._load_and_preprocess(int(patient_i))
        if float(mask_oh[..., 1:].sum()) <= 0.0:
            return None
        patches_img, _unused_mask, grid = self._patchify_all(img, mask_oh)
        return img, mask_oh, patches_img, grid


# =============================================================================
# Losses and metrics
# =============================================================================

def weighted_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, class_weights: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    n, c = y_true.shape[:2]
    yt = y_true.reshape(n, c, -1)
    yp = y_pred.reshape(n, c, -1)
    inter = torch.sum(yt * yp, dim=2)
    uni = torch.sum(yt + yp, dim=2)
    dice = (2.0 * inter + epsilon) / (uni + epsilon)
    w = class_weights.reshape(1, -1).to(dice.device)
    wd = torch.sum(w * dice, dim=1) / (torch.sum(w) + 1e-12)
    return 1.0 - torch.mean(wd)


def weighted_ce_from_logits(y_true_onehot: torch.Tensor, logits: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    target = torch.argmax(y_true_onehot.float(), dim=1).long()
    return F.cross_entropy(logits.float(), target, weight=class_weights.float(), reduction="mean")


def combined_loss_components_logits(
    y_true_onehot: torch.Tensor,
    logits: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = 0.7,
    epsilon: float = 1e-7,
) -> Tuple[torch.Tensor, float, float]:
    probs = torch.softmax(logits.float(), dim=1)
    wdice = weighted_dice_loss(y_true_onehot, probs, class_weights, epsilon)
    wce = weighted_ce_from_logits(y_true_onehot, logits, class_weights)
    total = alpha * wdice + (1.0 - alpha) * wce
    return total, float(wdice.detach().cpu().item()), float(wce.detach().cpu().item())


def weighted_cce_probs(
    y_true: torch.Tensor, y_pred: torch.Tensor, class_weights: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = y_pred.float()
    class_weights = class_weights.float()
    eps = max(float(epsilon), 1e-4)
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    cw = class_weights.reshape(1, -1, 1, 1, 1)
    return -torch.mean(torch.sum(cw * y_true * torch.log(y_pred), dim=1))


def combined_loss_components_probs(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = 0.7,
    epsilon: float = 1e-7,
) -> Tuple[torch.Tensor, float, float]:
    gdl = weighted_dice_loss(y_true, y_pred, class_weights, epsilon)
    wce = weighted_cce_probs(y_true, y_pred, class_weights, epsilon)
    total = alpha * gdl + (1.0 - alpha) * wce
    return total, float(gdl.detach().cpu().item()), float(wce.detach().cpu().item())


def hard_dice_metric(
    y_true: torch.Tensor, y_pred_probs: torch.Tensor, class_weights: torch.Tensor, epsilon: float = 1e-7
) -> float:
    y_true = y_true.float()
    y_pred_probs = y_pred_probs.float()
    _, c = y_true.shape[:2]
    if c <= 1:
        return 0.0
    yt = torch.argmax(y_true, dim=1)
    yp = torch.argmax(y_pred_probs, dim=1)
    weights_nobg = class_weights[1:].float()
    w_sum = float(weights_nobg.sum().detach().cpu().item())
    if w_sum <= 0:
        weights_nobg = torch.ones_like(weights_nobg)
        w_sum = float(weights_nobg.sum().detach().cpu().item())
    dices = []
    for cls in range(1, c):
        yt_c = yt == cls
        yp_c = yp == cls
        inter = (yt_c & yp_c).sum().float()
        denom = yt_c.sum().float() + yp_c.sum().float()
        dices.append((2.0 * inter + epsilon) / (denom + epsilon))
    stacked = torch.stack(dices, dim=0)
    weighted = (stacked * weights_nobg).sum() / (w_sum + 1e-12)
    return float(weighted.detach().cpu().item())


def unweighted_soft_dice_macro_fg(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-7) -> float:
    y_true = y_true.float()
    y_pred = y_pred.float()
    _, c = y_true.shape[:2]
    if c <= 1:
        return 0.0
    yt = y_true.reshape(1, c, -1)
    yp = y_pred.reshape(1, c, -1)
    inter = torch.sum(yt * yp, dim=2).squeeze(0)
    uni = torch.sum(yt + yp, dim=2).squeeze(0)
    dice = (2.0 * inter + epsilon) / (uni + epsilon)
    return float(torch.mean(dice[1:]).detach().cpu().item())


def unweighted_hard_dice_macro_fg(y_true: torch.Tensor, y_pred_probs: torch.Tensor, epsilon: float = 1e-7) -> float:
    y_true = y_true.float()
    _, c = y_true.shape[:2]
    if c <= 1:
        return 0.0
    yt = torch.argmax(y_true, dim=1)
    yp = torch.argmax(y_pred_probs.float(), dim=1)
    dices = []
    for cls in range(1, c):
        yt_c = yt == cls
        yp_c = yp == cls
        inter = (yt_c & yp_c).sum().float()
        denom = yt_c.sum().float() + yp_c.sum().float()
        dices.append((2.0 * inter + epsilon) / (denom + epsilon))
    return float(torch.mean(torch.stack(dices)).detach().cpu().item())


def dice_metric_weighted_soft_fg(y_true: torch.Tensor, y_pred: torch.Tensor, class_weights: torch.Tensor, epsilon: float = 1e-7) -> float:
    y_true = y_true.float()
    y_pred = y_pred.float()
    _, c = y_true.shape[:2]
    if c <= 1:
        return 0.0
    yt = y_true.reshape(1, c, -1)
    yp = y_pred.reshape(1, c, -1)
    inter = torch.sum(yt * yp, dim=2)
    uni = torch.sum(yt + yp, dim=2)
    dice = (2.0 * inter + epsilon) / (uni + epsilon)
    dice_fg = dice[:, 1:]
    w_fg = class_weights[1:].reshape(1, -1).to(dice_fg.device).float()
    return float((torch.sum(w_fg * dice_fg, dim=1) / (torch.sum(w_fg) + 1e-12)).mean().detach().cpu().item())


# =============================================================================
# Gaussian merge
# =============================================================================

def _gaussian_weight_3d(ph: int, pw: int, pd: int, sigma_scale: float = 0.125, eps: float = 1e-6) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, ph, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, pw, dtype=np.float32)
    zz = np.linspace(-1.0, 1.0, pd, dtype=np.float32)
    yy, xx, zz = np.meshgrid(yy, xx, zz, indexing="ij")
    sigma2 = max(sigma_scale * sigma_scale, 1e-6)
    w = np.exp(-0.5 * (yy * yy + xx * xx + zz * zz) / sigma2).astype(np.float32)
    w = w / (w.max() + eps)
    w = np.clip(w, eps, None)
    return w[..., np.newaxis]


def gaussian_merge_from_patchify_stack(
    model: nn.Module,
    device: torch.device,
    vol_img: np.ndarray,
    batch_patches: np.ndarray,
    grid: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int, int],
    num_classes: int,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    h, w, d = vol_img.shape[0], vol_img.shape[1], vol_img.shape[2]
    ph, pw, pd, cin = patch_size
    sh, sw, sd = int(step_size[0]), int(step_size[1]), int(step_size[2])
    nh, nw, nd = grid
    num_p = nh * nw * nd
    if batch_patches.shape[0] != num_p:
        raise ValueError(f"Patch count {batch_patches.shape[0]} != grid {grid} => {num_p}")

    weight_patch = _gaussian_weight_3d(ph, pw, pd)
    sum_probs = np.zeros((h, w, d, num_classes), dtype=np.float32)
    sum_w = np.zeros((h, w, d, 1), dtype=np.float32)

    for start in range(0, num_p, batch_size):
        end = min(start + batch_size, num_p)
        batch = batch_patches[start:end].astype(np.float32, copy=False)
        x = torch.from_numpy(batch).float().to(device).permute(0, 4, 3, 1, 2)
        with torch.no_grad():
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            probs = torch.softmax(logits.float(), dim=1)
            probs_np = probs.permute(0, 3, 4, 2, 1).contiguous().cpu().numpy().astype(np.float32)

        for local_i, p in enumerate(range(start, end)):
            ii, jj, kk = np.unravel_index(p, (nh, nw, nd))
            h0, w0, d0 = int(ii * sh), int(jj * sw), int(kk * sd)
            weighted = probs_np[local_i] * weight_patch
            sum_probs[h0 : h0 + ph, w0 : w0 + pw, d0 : d0 + pd, :] += weighted
            sum_w[h0 : h0 + ph, w0 : w0 + pw, d0 : d0 + pd, :] += weight_patch

    return (sum_probs / np.clip(sum_w, 1e-6, None)).astype(np.float32, copy=False)


# =============================================================================
# Pure 3D U-Net model
# =============================================================================

class InterpConv3d(nn.Module):
    """Upsample with trilinear interpolation, then project with Conv3d."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: Tuple[int, int, int]):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return self.proj(x)


class _ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.GELU(),
            nn.Dropout3d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """Baseline 3D U-Net (no Mamba)."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 16, dropout: float = 0.1):
        super().__init__()
        bc = base_channels

        self.enc1 = _ConvBlock3d(in_channels, bc, dropout)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = _ConvBlock3d(bc, bc * 2, dropout)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = _ConvBlock3d(bc * 2, bc * 4, dropout)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = _ConvBlock3d(bc * 4, bc * 8, dropout)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = _ConvBlock3d(bc * 8, bc * 16, dropout)

        self.up4 = InterpConv3d(bc * 16, bc * 8, scale_factor=(1, 2, 2))
        self.dec4 = _ConvBlock3d(bc * 16, bc * 8, dropout)

        self.up3 = InterpConv3d(bc * 8, bc * 4, scale_factor=(2, 2, 2))
        self.dec3 = _ConvBlock3d(bc * 8, bc * 4, dropout)

        self.up2 = InterpConv3d(bc * 4, bc * 2, scale_factor=(2, 2, 2))
        self.dec2 = _ConvBlock3d(bc * 4, bc * 2, dropout)

        self.up1 = InterpConv3d(bc * 2, bc, scale_factor=(2, 2, 2))
        self.dec1 = _ConvBlock3d(bc * 2, bc, dropout)

        self.out_conv = nn.Conv3d(bc, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="trilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


# =============================================================================
# Training / evaluation
# =============================================================================

def get_paths(root: str) -> Dict[str, str]:
    return {
        "train_img": os.path.join(root, "train_images"),
        "train_mask": os.path.join(root, "train_masks"),
        "val_img": os.path.join(root, "val_images"),
        "val_mask": os.path.join(root, "val_masks"),
        "test_img": os.path.join(root, "test_images"),
        "test_mask": os.path.join(root, "test_masks"),
    }


def cyclic_lr(step: int) -> float:
    cycle = math.floor(1 + step / (2 * CLR_STEP_SIZE))
    x = abs(step / CLR_STEP_SIZE - 2 * cycle + 1)
    return float(INITIAL_LR + (MAX_LR - INITIAL_LR) * max(0.0, 1 - x))


def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def run_merged_eval(
    model: nn.Module,
    device: torch.device,
    gen: PatchGenerator,
    class_weights: torch.Tensor,
    patch_size: Tuple[int, int, int, int],
    step_size: Tuple[int, int, int],
    max_patients: Optional[int],
    merge_batch_size: int = 8,
) -> Optional[dict]:
    n = len(gen.image_files)
    if max_patients is not None:
        n = min(n, int(max_patients))

    tot_loss = 0.0
    tot_gdl = 0.0
    tot_wce = 0.0
    tot_wdice_fg = 0.0
    tot_hard = 0.0
    tot_uwd = 0.0
    tot_uhd = 0.0
    n_done = 0

    for i in tqdm(range(n), desc="Merged eval", leave=False):
        out = gen.get_volume_for_merge(i)
        if out is None:
            continue
        img_full, mask_oh, patches_img, grid = out
        probs_full = gaussian_merge_from_patchify_stack(
            model=model,
            device=device,
            vol_img=img_full,
            batch_patches=patches_img,
            grid=grid,
            step_size=step_size,
            patch_size=patch_size,
            num_classes=len(CLASS_NAMES),
            batch_size=merge_batch_size,
        )
        y = torch.from_numpy(mask_oh[np.newaxis]).float().to(device).permute(0, 4, 3, 1, 2)
        p = torch.from_numpy(probs_full[np.newaxis]).float().to(device).permute(0, 4, 3, 1, 2)
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

        loss_t, gdl_v, wce_v = combined_loss_components_probs(y, p, class_weights)
        wdice_fg = dice_metric_weighted_soft_fg(y, p, class_weights)
        hdm = hard_dice_metric(y, p, class_weights)
        uwd = unweighted_soft_dice_macro_fg(y, p)
        uhd = unweighted_hard_dice_macro_fg(y, p)

        tot_loss += float(loss_t.detach().cpu().item())
        tot_gdl += gdl_v
        tot_wce += wce_v
        tot_wdice_fg += wdice_fg
        tot_hard += hdm
        tot_uwd += uwd
        tot_uhd += uhd
        n_done += 1

    if n_done == 0:
        return None

    return {
        "n_patients": n_done,
        "avg_val_loss": tot_loss / n_done,
        "avg_gdl": tot_gdl / n_done,
        "avg_wce": tot_wce / n_done,
        "avg_weighted_soft_dice_fg": tot_wdice_fg / n_done,
        "avg_hard_dice_metric": tot_hard / n_done,
        "avg_unweighted_soft_dice_macro_fg": tot_uwd / n_done,
        "avg_unweighted_hard_dice_macro_fg": tot_uhd / n_done,
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    paths = get_paths(DATASET_ROOT)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    num_classes = len(CLASS_NAMES)
    cin = len(CT_WINDOWS)
    patch_size = (PATCH_H, PATCH_W, PATCH_D, cin)

    train_gen = PatchGenerator(paths["train_img"], paths["train_mask"], patch_size, TRAIN_STEP, MAX_PATIENTS, True, True, FOREGROUND_THRESHOLD, num_classes, CT_WINDOWS)
    val_gen = PatchGenerator(paths["val_img"], paths["val_mask"], patch_size, VAL_STEP, MAX_PATIENTS, False, False, -1.0, num_classes, CT_WINDOWS)
    test_gen = PatchGenerator(paths["test_img"], paths["test_mask"], patch_size, INFER_STEP, MAX_PATIENTS, False, False, -1.0, num_classes, CT_WINDOWS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet3D(in_channels=cin, num_classes=num_classes, base_channels=BASE_CHANNELS, dropout=0.1).to(device)
    print(f"Model on {device} (base_channels={BASE_CHANNELS}, architecture=UNet3D baseline)")
    class_weights = torch.tensor(CLASS_WEIGHT_VALUES, dtype=torch.float32, device=device)
    print(f"Dampened weights: {CLASS_WEIGHT_VALUES}")
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.pkl")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_gdl": [],
        "val_wce": [],
        "val_weighted_soft_dice_fg": [],
        "val_hard_dice_metric": [],
        "val_unweighted_soft_dice_macro_fg": [],
        "val_unweighted_hard_dice_macro_fg": [],
        "lr": [],
        "epoch_time_s": [],
    }
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    if TRAINING_MODE == "continue" and os.path.exists(latest_ckpt):
        ck = load_checkpoint(latest_ckpt, device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        patience_counter = int(ck.get("patience_counter", 0))
        history = ck.get("history", history)
        vl = history.get("val_loss") or []
        best_from_history = min(vl) if vl else float("inf")
        best_from_ckpt = float(ck["best_val_loss"]) if "best_val_loss" in ck else float("inf")
        best_val_loss = min(best_from_ckpt, best_from_history)
        global_step = int(ck.get("global_step", 0))
        if scaler is not None and "scaler_state_dict" in ck:
            scaler.load_state_dict(ck["scaler_state_dict"])
        print(f"Resumed from epoch {start_epoch} (best val_loss={best_val_loss:.4f}).")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for patient_i in tqdm(range(len(train_gen)), desc=f"Epoch {epoch+1}/{EPOCHS} train", leave=False):
            vb = train_gen.get_train_patches(patient_i)
            if vb.patches_img.shape[0] == 0:
                continue
            for pi in range(vb.patches_img.shape[0]):
                lr = cyclic_lr(global_step)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)

                x_np = vb.patches_img[pi : pi + 1]
                y_np = vb.patches_mask[pi : pi + 1]
                x = torch.from_numpy(x_np).float().to(device).permute(0, 4, 3, 1, 2)
                y = torch.from_numpy(y_np).float().to(device).permute(0, 4, 3, 1, 2)

                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x)
                        loss, _wd, _wce = combined_loss_components_logits(y, logits, class_weights)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss, _wd, _wce = combined_loss_components_logits(y, logits, class_weights)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss_sum += float(loss.detach().cpu().item())
                train_steps += 1
                global_step += 1

                del x, y, logits, loss
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gc.collect()

        avg_train_loss = train_loss_sum / max(1, train_steps)

        val_metrics = run_merged_eval(model, device, val_gen, class_weights, patch_size, VAL_STEP, None, merge_batch_size=8)
        if val_metrics is None:
            val_loss = float("nan")
            val_gdl = val_wce = val_wdice = val_hard = val_uwd = val_uhd = 0.0
        else:
            val_loss = float(val_metrics["avg_val_loss"])
            val_gdl = float(val_metrics["avg_gdl"])
            val_wce = float(val_metrics["avg_wce"])
            val_wdice = float(val_metrics["avg_weighted_soft_dice_fg"])
            val_hard = float(val_metrics["avg_hard_dice_metric"])
            val_uwd = float(val_metrics["avg_unweighted_soft_dice_macro_fg"])
            val_uhd = float(val_metrics["avg_unweighted_hard_dice_macro_fg"])

        t_epoch = time.time() - t0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_gdl"].append(val_gdl)
        history["val_wce"].append(val_wce)
        history["val_weighted_soft_dice_fg"].append(val_wdice)
        history["val_hard_dice_metric"].append(val_hard)
        history["val_unweighted_soft_dice_macro_fg"].append(val_uwd)
        history["val_unweighted_hard_dice_macro_fg"].append(val_uhd)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["epoch_time_s"].append(float(t_epoch))

        is_best = val_loss == val_loss and val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "history": history,
            "global_step": global_step,
        }
        if scaler is not None:
            ckpt_payload["scaler_state_dict"] = scaler.state_dict()
        save_checkpoint(latest_ckpt, ckpt_payload)
        if is_best:
            save_checkpoint(best_ckpt, ckpt_payload)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        print(
            f"Epoch {epoch+1:04d} | train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_wDice_fg={val_wdice:.4f} | val_hardDice={val_hard:.4f} | "
            f"val_unw_soft={val_uwd:.4f} | val_unw_hard={val_uhd:.4f} | "
            f"best_val_loss={best_val_loss:.4f} | patience={patience_counter}/{PATIENCE} | time={t_epoch:.1f}s"
        )
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

        train_gen.on_epoch_end()

    test_metrics = run_merged_eval(model, device, test_gen, class_weights, patch_size, INFER_STEP, None, merge_batch_size=8)
    print("Test metrics (merged volume, INFER_STEP / 50% overlap H,W,D):", test_metrics)


if __name__ == "__main__":
    main()

