# -*- coding: utf-8 -*-
"""
Teaching template: 3D Tri-Oriented Mamba U-Net for multi-class 3D segmentation.

This script is intended to be uploaded to GitHub as a public, re-trainable template.
It contains no personal file paths and no test-time augmentation (TTA).

## Dataset layout

DATASET_ROOT/
  train_images/   train_masks/
  val_images/     val_masks/
  test_images/    test_masks/

Images/masks are expected to be NIfTI `.nii` volumes.

## Shapes

- NumPy volumes/patches: (H, W, D, C)
- PyTorch tensors:       (N, C, D, H, W)

## Evaluation

Merged-volume metrics stitch patch softmax predictions with a Gaussian blend.
**Validation** uses `VAL_STEP` (50% overlap on H/W; depth stride equals patch depth, so no overlap along D).
**Test** uses `INFER_STEP` (50% overlap on H, W, and D). This split matches the reference bachelor Colab setup.

The reported test Dice metrics are **unweighted** (macro-average across foreground classes).
Training losses can remain class-weighted (weighted Dice + weighted CE) to handle imbalance.
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
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints_mamba")

# If you want to resume, set TRAINING_MODE="continue". Otherwise start from scratch.
TRAINING_MODE = os.environ.get("TRAINING_MODE", "new")  # "new" | "continue"

# Volume preprocessing / tiling
HEIGHT = 512
WIDTH = 512
PATCH_D = 8

# Patch size in (H, W, D, C_in)
# Default uses 4 CT windows (brain, bone, subdural, stroke).
PATCH_H = HEIGHT // 4  # 128
PATCH_W = WIDTH // 4   # 128

# Validation merge: 50% overlap H/W; depth stride = patch depth (no overlap along D).
VAL_STEP = (PATCH_H // 2, PATCH_W // 2, PATCH_D)  # e.g. (64, 64, 8) when PATCH_D=8

# Test / final eval: 50% overlap on H, W, and D.
INFER_STEP = (PATCH_H // 2, PATCH_W // 2, PATCH_D // 2)  # e.g. (64, 64, 4)

# Training step can be less overlapped (fewer patches / faster).
TRAIN_STEP = (PATCH_H, PATCH_W, PATCH_D // 2)  # (128, 128, 4)

BATCH_SIZE_PATIENTS = 1  # one patient at a time; patches are iterated inside the loop
EPOCHS = 200
PATIENCE = 20
MAX_PATIENTS: Optional[int] = None  # e.g. 20 for quick demo; None = all

# Augmentation (applied per volume with probability AUG_PROB)
AUG_PROB = 0.5

# Patch filtering: drop background-only patches during training (0 keeps any foreground)
FOREGROUND_THRESHOLD = 0.0

# Cyclic learning rate (triangular)
INITIAL_LR = 1e-5
MAX_LR = 3e-4
CLR_STEP_SIZE = 2290

# Optimizer
WEIGHT_DECAY = 1e-5

# Classes: background + 5 hemorrhage types (edit names if your label map differs)
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
}

# CT windows: [center, width]
CT_WINDOWS = [
    [40, 80],     # brain
    [600, 3000],  # bone
    [75, 215],    # subdural
    [60, 120],    # stroke/hemorrhage-ish
]

# Class weights (optional): used for weighted losses (Dice + CE). Foreground upweighted by default.
# Replace these with weights computed on your dataset if available.
CLASS_WEIGHT_VALUES = [1.0] + [4.0] * (len(CLASS_NAMES) - 1)


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
        return np.zeros_like(image, dtype=np.float32)
    return ((image - mn) / (mx - mn)).astype(np.float32, copy=False)


def one_hot_np(mask: np.ndarray, num_classes: int) -> np.ndarray:
    mask = mask.astype(np.int64, copy=False)
    out = np.eye(num_classes, dtype=np.float32)[mask]
    return out


def pad_depth_axis_background(img: np.ndarray, mask_oh: np.ndarray, pad_amount: int) -> Tuple[np.ndarray, np.ndarray]:
    if pad_amount <= 0:
        return img, mask_oh
    h, w, d, c = img.shape
    _, _, _, k = mask_oh.shape
    img_tail = np.zeros((h, w, pad_amount, c), dtype=np.float32)
    mask_tail = np.zeros((h, w, pad_amount, k), dtype=np.float32)
    mask_tail[..., 0] = 1.0
    return np.concatenate([img, img_tail], axis=2), np.concatenate([mask_oh, mask_tail], axis=2)


def zoom_2d(img: np.ndarray, mask: np.ndarray, zoom_factor: float) -> Tuple[np.ndarray, np.ndarray]:
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

    def _fit_spatial(vol: np.ndarray) -> np.ndarray:
        out = np.asarray(vol)
        hh, ww = out.shape[0], out.shape[1]
        if hh > h:
            top = (hh - h) // 2
            out = out[top : top + h, :, :, :]
        if ww > w:
            left = (ww - w) // 2
            out = out[:, left : left + w, :, :]
        hh, ww = out.shape[0], out.shape[1]
        if hh < h or ww < w:
            pad_top = (h - hh) // 2
            pad_bottom = h - hh - pad_top
            pad_left = (w - ww) // 2
            pad_right = w - ww - pad_left
            try:
                out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0)), mode="reflect")
            except ValueError:
                out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
        return out

    return _fit_spatial(img_scaled).astype(np.float32, copy=False), _fit_spatial(mask_scaled).astype(np.float32, copy=False)


def mirroring(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # H flip
    if random.random() < 0.5:
        img = np.flip(img, axis=0)
        mask = np.flip(mask, axis=0)
    # W flip
    if random.random() < 0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
    return img, mask


def random_rotate_90(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = random.choice([0, 1, 2, 3])
    if k == 0:
        return img, mask
    img_out = np.empty_like(img)
    mask_out = np.empty_like(mask)
    for zz in range(img.shape[2]):
        img_out[:, :, zz, :] = np.rot90(img[:, :, zz, :], k=k, axes=(0, 1))
        mask_out[:, :, zz, :] = np.rot90(mask[:, :, zz, :], k=k, axes=(0, 1))
    return img_out, mask_out


@dataclass
class VolumeBatch:
    patches_img: np.ndarray  # (P, ph, pw, pd, Cin)
    patches_mask: np.ndarray  # (P, ph, pw, pd, C)


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

    def _load_and_preprocess(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(self.image_files[idx], mmap=True).get_fdata(dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        mask = nib.load(self.mask_files[idx], mmap=True).get_fdata().astype(np.int64, copy=False)

        img = normalize_image01(standardize_windows(img, self.windows))
        mask_oh = one_hot_np(mask, self.num_classes).astype(np.float32, copy=False)  # (H,W,D,C)

        # Resize to canonical size
        img = resize(
            img,
            (HEIGHT, WIDTH, img.shape[2], img.shape[3]),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)
        mask_oh = resize(
            mask_oh,
            (HEIGHT, WIDTH, mask_oh.shape[2], self.num_classes),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32, copy=False)

        if self.augment and random.random() < AUG_PROB:
            zf = random.uniform(0.7, 1.3)
            img, mask_oh = zoom_2d(img, mask_oh, zf)
            img, mask_oh = mirroring(img, mask_oh)
            img, mask_oh = random_rotate_90(img, mask_oh)

        return img, mask_oh

    def _patchify_all(self, img: np.ndarray, mask_oh: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        ph, pw, pd, cin = self.patch_size
        sh, sw, sd = self.step_size

        # Depth pad so tiling grid fits
        remainder = (img.shape[2] - pd) % sd
        if remainder != 0:
            pad_amount = sd - remainder
            img, mask_oh = pad_depth_axis_background(img, mask_oh, pad_amount)

        patches_img = patchify(img, (ph, pw, pd, cin), step=(sh, sw, sd, cin))
        patches_mask = patchify(mask_oh, (ph, pw, pd, self.num_classes), step=(sh, sw, sd, self.num_classes))
        nh, nw, nd = patches_img.shape[0], patches_img.shape[1], patches_img.shape[2]
        patches_img = patches_img.reshape(-1, ph, pw, pd, cin).astype(np.float32, copy=False)
        patches_mask = patches_mask.reshape(-1, ph, pw, pd, self.num_classes).astype(np.float32, copy=False)
        return patches_img, patches_mask, (nh, nw, nd)

    def get_train_patches(self, patient_i: int) -> VolumeBatch:
        idx = int(self.indices[patient_i])
        img, mask_oh = self._load_and_preprocess(idx)

        # skip empty volumes (no foreground)
        if float(mask_oh[..., 1:].sum()) <= 0.0:
            return VolumeBatch(patches_img=np.empty((0,) + self.patch_size, np.float32), patches_mask=np.empty((0, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.num_classes), np.float32))

        patches_img, patches_mask, _grid = self._patchify_all(img, mask_oh)

        fg_vox = patches_mask[..., 1:].sum(axis=(1, 2, 3, 4))
        keep = (fg_vox / max(1.0, float(self.patch_voxels))) > self.threshold
        return VolumeBatch(patches_img=patches_img[keep], patches_mask=patches_mask[keep])

    def get_volume_for_merge(self, patient_i: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]]:
        idx = int(patient_i)
        img, mask_oh = self._load_and_preprocess(idx)
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
    """Weighted cross-entropy on softmax probabilities (one-hot targets)."""
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
    """Combined loss on probabilities: alpha * GDL + (1-alpha) * weighted CE (e.g. merged-volume eval)."""
    gdl = weighted_dice_loss(y_true, y_pred, class_weights, epsilon)
    wce = weighted_cce_probs(y_true, y_pred, class_weights, epsilon)
    total = alpha * gdl + (1.0 - alpha) * wce
    return total, float(gdl.detach().cpu().item()), float(wce.detach().cpu().item())


def hard_dice_metric(
    y_true: torch.Tensor, y_pred_probs: torch.Tensor, class_weights: torch.Tensor, epsilon: float = 1e-7
) -> float:
    """Weighted hard (argmax) Dice over foreground classes."""
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
# Gaussian merge (overlapping patch stitching)
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
    return w[..., np.newaxis]  # (ph, pw, pd, 1)


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
        x = torch.from_numpy(batch).float().to(device).permute(0, 4, 3, 1, 2)  # (B,C,D,H,W)
        with torch.no_grad():
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            probs = torch.softmax(logits.float(), dim=1)
            probs_np = probs.permute(0, 3, 4, 2, 1).contiguous().cpu().numpy().astype(np.float32)  # (B,ph,pw,pd,C)

        for local_i, p in enumerate(range(start, end)):
            ii, jj, kk = np.unravel_index(p, (nh, nw, nd))
            h0, w0, d0 = int(ii * sh), int(jj * sw), int(kk * sd)
            weighted = probs_np[local_i] * weight_patch
            sum_probs[h0 : h0 + ph, w0 : w0 + pw, d0 : d0 + pd, :] += weighted
            sum_w[h0 : h0 + ph, w0 : w0 + pw, d0 : d0 + pd, :] += weight_patch

    out = sum_probs / np.clip(sum_w, 1e-6, None)
    return out.astype(np.float32, copy=False)


# =============================================================================
# Mamba SSM + Tri-Oriented blocks + 3D U-Net
# =============================================================================

def _parallel_scan_linear(A_bar: torch.Tensor, B_bar_u: torch.Tensor) -> torch.Tensor:
    n, l, d_inner, d_state = A_bar.shape
    if l == 0:
        return B_bar_u

    device = A_bar.device
    dtype = A_bar.dtype
    block_len = 2048
    h_prev = torch.zeros(n, d_inner, d_state, device=device, dtype=dtype)
    out = torch.empty(n, l, d_inner, d_state, device=device, dtype=dtype)

    for start in range(0, l, block_len):
        end = min(start + block_len, l)
        Ab = A_bar[:, start:end]
        Bb = B_bar_u[:, start:end]
        lb = end - start

        l_pad = 1 << (lb - 1).bit_length()
        seg_A = torch.ones(n, l_pad, d_inner, d_state, device=device, dtype=dtype)
        seg_B = torch.zeros(n, l_pad, d_inner, d_state, device=device, dtype=dtype)
        seg_A[:, :lb] = Ab
        seg_B[:, :lb] = Bb

        num_steps = max(0, (l_pad - 1).bit_length())
        for k in range(num_steps):
            step = 1 << k
            if step >= l_pad:
                break
            old_A = seg_A[:, step:]
            new_A = seg_A[:, : l_pad - step] * old_A
            new_B = old_A * seg_B[:, : l_pad - step] + seg_B[:, step:]
            seg_A = torch.cat([seg_A[:, :step], new_A], dim=1)
            seg_B = torch.cat([seg_B[:, :step], new_B], dim=1)

        a_pref = seg_A[:, :lb]
        b_pref = seg_B[:, :lb]
        h_block = a_pref * h_prev.unsqueeze(1) + b_pref
        out[:, start:end] = h_block
        h_prev = h_block[:, -1]

    return out


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.x_proj_delta = nn.Linear(self.d_inner, self.d_inner)
        self.x_proj_b = nn.Linear(self.d_inner, self.d_inner * d_state)
        self.x_proj_c = nn.Linear(self.d_inner, self.d_inner * d_state)
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, l, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        x_conv = x_part.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv)[:, :, -l:]
        x_conv = F.silu(x_conv.permute(0, 2, 1))

        delta = F.softplus(self.x_proj_delta(x_conv))
        B = self.x_proj_b(x_conv).reshape(n, l, self.d_inner, self.d_state)
        C = self.x_proj_c(x_conv).reshape(n, l, self.d_inner, self.d_state)

        A = -torch.exp(self.A_log.to(dtype=x_conv.dtype))
        delta = delta.to(dtype=x_conv.dtype)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        one = A_bar.new_tensor(1.0)
        eps = A_bar.new_tensor(1e-6)
        B_bar = (A_bar - one) / (A.unsqueeze(0).unsqueeze(0) + eps) * B
        B_bar = torch.nan_to_num(B_bar, nan=0.0, posinf=0.0, neginf=0.0)
        B_bar_u = B_bar * x_conv.unsqueeze(-1)
        h = _parallel_scan_linear(A_bar, B_bar_u)
        y_ssm = (C.float() * h).sum(-1)

        y = y_ssm + x_conv.float() * self.D.unsqueeze(0).unsqueeze(0)
        y = y.clamp(-200.0, 200.0)
        z_f = z.float().clamp(-10.0, 10.0)
        gate = y * F.silu(z_f)
        out = self.out_proj(gate)
        return out


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(x)
        return x + self.ff(x)


class TriOrientedMambaBlock_NoExt(nn.Module):
    def __init__(self, channels: int, ssm_dim: int = 64, dropout_rate: float = 0.1, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba_d = MambaBlock(d_model=channels, d_state=ssm_dim, d_conv=d_conv, expand=expand)
        self.mamba_h = MambaBlock(d_model=channels, d_state=ssm_dim, d_conv=d_conv, expand=expand)
        self.mamba_w = MambaBlock(d_model=channels, d_state=ssm_dim, d_conv=d_conv, expand=expand)
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.act = nn.GELU()

    def _scan_d(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        seq = x.permute(0, 2, 3, 4, 1).reshape(n, d * h * w, c)
        out = self.mamba_d(seq)
        return out.reshape(n, d, h, w, c).permute(0, 4, 1, 2, 3)

    def _scan_h(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        x_h = x.permute(0, 1, 3, 2, 4).contiguous()
        seq = x_h.permute(0, 2, 3, 4, 1).reshape(n, h * d * w, c)
        out = self.mamba_h(seq)
        return out.reshape(n, h, d, w, c).permute(0, 4, 2, 1, 3).contiguous()

    def _scan_w(self, x: torch.Tensor) -> torch.Tensor:
        n, c, d, h, w = x.shape
        x_w = x.permute(0, 1, 4, 2, 3).contiguous()
        seq = x_w.permute(0, 2, 3, 4, 1).reshape(n, w * d * h, c)
        out = self.mamba_w(seq)
        return out.reshape(n, w, d, h, c).permute(0, 4, 2, 3, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (self._scan_d(x) + self._scan_h(x) + self._scan_w(x)) / 3.0
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return x + y


class InterpConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: Tuple[int, int, int]):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return self.proj(x)


class Mamba3DUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 16, ssm_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        TriBlock = TriOrientedMambaBlock_NoExt

        self.enc1_conv = nn.Sequential(nn.Conv3d(in_channels, base_channels, 3, padding=1), nn.InstanceNorm3d(base_channels), nn.GELU())
        self.enc1_mamba_1 = TriBlock(base_channels, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2_conv = nn.Sequential(nn.Conv3d(base_channels, base_channels * 2, 3, padding=1), nn.InstanceNorm3d(base_channels * 2), nn.GELU())
        self.enc2_mamba_1 = TriBlock(base_channels * 2, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.enc2_mamba_2 = TriBlock(base_channels * 2, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3_conv = nn.Sequential(nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1), nn.InstanceNorm3d(base_channels * 4), nn.GELU())
        self.enc3_mamba_1 = TriBlock(base_channels * 4, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.enc3_mamba_2 = TriBlock(base_channels * 4, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4_conv = nn.Sequential(nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1), nn.InstanceNorm3d(base_channels * 8), nn.GELU())
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck_conv = nn.Sequential(nn.Conv3d(base_channels * 8, base_channels * 16, 3, padding=1), nn.InstanceNorm3d(base_channels * 16), nn.GELU())

        self.up4 = InterpConv3d(base_channels * 16, base_channels * 8, scale_factor=(1, 2, 2))
        self.dec4_conv = nn.Sequential(nn.Conv3d(base_channels * 16, base_channels * 8, 3, padding=1), nn.InstanceNorm3d(base_channels * 8), nn.GELU())

        self.up3 = InterpConv3d(base_channels * 8, base_channels * 4, scale_factor=(2, 2, 2))
        self.dec3_conv = nn.Sequential(nn.Conv3d(base_channels * 8, base_channels * 4, 3, padding=1), nn.InstanceNorm3d(base_channels * 4), nn.GELU())
        self.dec3_mamba_1 = TriBlock(base_channels * 4, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.dec3_mamba_2 = TriBlock(base_channels * 4, ssm_dim=ssm_dim, dropout_rate=dropout)

        self.up2 = InterpConv3d(base_channels * 4, base_channels * 2, scale_factor=(2, 2, 2))
        self.dec2_conv = nn.Sequential(nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1), nn.InstanceNorm3d(base_channels * 2), nn.GELU())
        self.dec2_mamba_1 = TriBlock(base_channels * 2, ssm_dim=ssm_dim, dropout_rate=dropout)
        self.dec2_mamba_2 = TriBlock(base_channels * 2, ssm_dim=ssm_dim, dropout_rate=dropout)

        self.up1 = InterpConv3d(base_channels * 2, base_channels, scale_factor=(2, 2, 2))
        self.dec1_conv = nn.Sequential(nn.Conv3d(base_channels * 2, base_channels, 3, padding=1), nn.InstanceNorm3d(base_channels), nn.GELU())
        self.dec1_mamba_1 = TriBlock(base_channels, ssm_dim=ssm_dim, dropout_rate=dropout)

        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1_conv(x)
        e1 = e1 + self.enc1_mamba_1(e1)
        p1 = self.pool1(e1)

        e2 = self.enc2_conv(p1)
        e2 = e2 + self.enc2_mamba_1(e2)
        e2 = e2 + self.enc2_mamba_2(e2)
        p2 = self.pool2(e2)

        e3 = self.enc3_conv(p2)
        e3 = e3 + self.enc3_mamba_1(e3)
        e3 = e3 + self.enc3_mamba_2(e3)
        p3 = self.pool3(e3)

        e4 = self.enc4_conv(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck_conv(p4)

        d4 = self.up4(b)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="trilinear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_conv(d4)

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv(d3)
        d3 = d3 + self.dec3_mamba_1(d3)
        d3 = d3 + self.dec3_mamba_2(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)
        d2 = d2 + self.dec2_mamba_1(d2)
        d2 = d2 + self.dec2_mamba_2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)
        d1 = d1 + self.dec1_mamba_1(d1)
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

    train_gen = PatchGenerator(
        image_dir=paths["train_img"],
        mask_dir=paths["train_mask"],
        patch_size=patch_size,
        step_size=TRAIN_STEP,
        max_patients=MAX_PATIENTS,
        augment=True,
        shuffle=True,
        threshold=FOREGROUND_THRESHOLD,
        num_classes=num_classes,
        windows=CT_WINDOWS,
    )
    val_gen = PatchGenerator(
        image_dir=paths["val_img"],
        mask_dir=paths["val_mask"],
        patch_size=patch_size,
        step_size=VAL_STEP,
        max_patients=MAX_PATIENTS,
        augment=False,
        shuffle=False,
        threshold=-1.0,
        num_classes=num_classes,
        windows=CT_WINDOWS,
    )
    test_gen = PatchGenerator(
        image_dir=paths["test_img"],
        mask_dir=paths["test_mask"],
        patch_size=patch_size,
        step_size=INFER_STEP,
        max_patients=MAX_PATIENTS,
        augment=False,
        shuffle=False,
        threshold=-1.0,
        num_classes=num_classes,
        windows=CT_WINDOWS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _base = 16
    _ssm = 16
    model = Mamba3DUNet(in_channels=cin, num_classes=num_classes, base_channels=_base, ssm_dim=_ssm, dropout=0.1).to(device)
    print(f"Model: Mamba3DUNet(cin={cin}, classes={num_classes}, base={_base}, ssm_dim={_ssm})")

    class_weights = torch.tensor(CLASS_WEIGHT_VALUES, dtype=torch.float32, device=device)

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

        # Validation: merged volume with VAL_STEP tiling (Gaussian stitch)
        val_metrics = run_merged_eval(
            model=model,
            device=device,
            gen=val_gen,
            class_weights=class_weights,
            patch_size=patch_size,
            step_size=VAL_STEP,
            max_patients=None,
            merge_batch_size=8,
        )
        if val_metrics is None:
            print("Validation produced no foreground volumes; skipping early-stopping update.")
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

    # Final test evaluation (merged, no TTA)
    test_metrics = run_merged_eval(
        model=model,
        device=device,
        gen=test_gen,
        class_weights=class_weights,
        patch_size=patch_size,
        step_size=INFER_STEP,
        max_patients=None,
        merge_batch_size=8,
    )
    print("Test metrics (merged volume, INFER_STEP / 50% overlap H,W,D):", test_metrics)


if __name__ == "__main__":
    main()

