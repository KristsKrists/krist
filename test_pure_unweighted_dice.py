# -*- coding: utf-8 -*-
"""
Evaluate a trained UNet3D (pure) checkpoint on the BHSD **test** split.

Uses the same pipeline as pure3d_unet_template.py end-of-training test:
  PatchGenerator(test) -> Gaussian merge -> INFER_STEP (50% overlap H,W,D)

Primary metrics (paper-comparable): **unweighted** macro foreground Dice (soft + hard).
Results are written to a .txt file in this directory.

Usage:
  cd github_templates
  python test_pure_unweighted_dice.py --checkpoint C:\\path\\to\\best_model.pt --data-root C:\\path\\to\\BHSD_split

If --checkpoint or --data-root are omitted, you will be prompted.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TEMPLATE_FILE = "pure3d_unet_template.py"
DEFAULT_OUTPUT = HERE / "pure_test_unweighted_dice_results.txt"


def _load_template_module() -> Any:
    path = HERE / TEMPLATE_FILE
    spec = importlib.util.spec_from_file_location("_pure_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_size(mod: Any) -> Tuple[int, int, int, int]:
    return (mod.PATCH_H, mod.PATCH_W, mod.PATCH_D, len(mod.CT_WINDOWS))


def _test_layout_ok(data_root: Path) -> bool:
    for sub in ("test_images", "test_masks"):
        d = data_root / sub
        if not d.is_dir() or not any(d.glob("*.nii")):
            return False
    return True


def _prompt_path(label: str, default: Optional[str] = None) -> Path:
    hint = f" [{default}]" if default else ""
    raw = input(f"{label}{hint}: ").strip().strip('"')
    if not raw:
        if default is None:
            raise SystemExit(f"{label} is required.")
        raw = default
    return Path(raw).expanduser().resolve()


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model_state_dict", ck)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {ckpt_path}")
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "epoch": ck.get("epoch"),
        "best_val_loss": ck.get("best_val_loss"),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _format_metrics_txt(
    model_name: str,
    ckpt_path: Path,
    data_root: Path,
    step_size: Tuple[int, int, int],
    load_info: Dict[str, Any],
    metrics: Dict[str, float],
    n_patients: int,
) -> str:
    soft = metrics["avg_unweighted_soft_dice_macro_fg"]
    hard = metrics["avg_unweighted_hard_dice_macro_fg"]
    lines = [
        f"{model_name} — BHSD TEST — unweighted Dice (merged volume)",
        "=" * 72,
        f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Checkpoint: {ckpt_path}",
        f"Dataset root: {data_root}",
        f"Split: test_images / test_masks",
        f"Step (INFER_STEP): {step_size}",
        f"Patients with foreground: {n_patients}",
        "",
        "Checkpoint metadata:",
        f"  epoch (saved): {load_info.get('epoch')}",
        f"  best_val_loss (saved): {load_info.get('best_val_loss')}",
        f"  missing_keys: {len(load_info.get('missing_keys', []))}",
        f"  unexpected_keys: {len(load_info.get('unexpected_keys', []))}",
    ]
    if load_info.get("missing_keys"):
        lines.append("  missing (first 10): " + ", ".join(load_info["missing_keys"][:10]))
    if load_info.get("unexpected_keys"):
        lines.append("  unexpected (first 10): " + ", ".join(load_info["unexpected_keys"][:10]))
    lines.extend(
        [
            "",
            "Primary metrics (unweighted macro foreground — comparable to papers):",
            f"  unweighted soft Dice: {soft:.6f}  ({100.0 * soft:.2f}%)",
            f"  unweighted hard Dice: {hard:.6f}  ({100.0 * hard:.2f}%)",
            "",
            "Other merged-volume metrics (from template run_merged_eval):",
            f"  avg_val_loss (combined): {metrics['avg_val_loss']:.6f}",
            f"  avg_gdl: {metrics['avg_gdl']:.6f}",
            f"  avg_wce: {metrics['avg_wce']:.6f}",
            f"  avg_weighted_soft_dice_fg: {metrics['avg_weighted_soft_dice_fg']:.6f}",
            f"  avg_hard_dice_metric (weighted): {metrics['avg_hard_dice_metric']:.6f}",
            "",
        ]
    )
    return "\n".join(lines)


def run_eval(
    checkpoint: Path,
    data_root: Path,
    output: Path,
    max_patients: Optional[int],
    merge_batch_size: int,
) -> None:
    mod = _load_template_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available; full-volume merge eval will be very slow on CPU.")

    paths = mod.get_paths(str(data_root))
    patch_size = _patch_size(mod)
    test_gen = mod.PatchGenerator(
        image_dir=paths["test_img"],
        mask_dir=paths["test_mask"],
        patch_size=patch_size,
        step_size=mod.INFER_STEP,
        max_patients=mod.MAX_PATIENTS,
        augment=False,
        shuffle=False,
        threshold=-1.0,
        num_classes=len(mod.CLASS_NAMES),
        windows=mod.CT_WINDOWS,
    )

    model = mod.UNet3D(
        in_channels=len(mod.CT_WINDOWS),
        num_classes=len(mod.CLASS_NAMES),
        base_channels=mod.BASE_CHANNELS,
        dropout=0.1,
    ).to(device)
    model.eval()

    load_info = _load_checkpoint(model, checkpoint, device)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(
            "Warning: checkpoint keys do not match template exactly "
            f"(missing={len(load_info['missing_keys'])}, unexpected={len(load_info['unexpected_keys'])}). "
            "If scores look wrong, check BASE_CHANNELS vs training Colab."
        )

    class_weights = torch.tensor(mod.CLASS_WEIGHT_VALUES, dtype=torch.float32, device=device)
    metrics = mod.run_merged_eval(
        model=model,
        device=device,
        gen=test_gen,
        class_weights=class_weights,
        patch_size=patch_size,
        step_size=mod.INFER_STEP,
        max_patients=max_patients,
        merge_batch_size=merge_batch_size,
    )
    if metrics is None:
        raise RuntimeError("No test patients with foreground hemorrhage; cannot compute Dice.")

    text = _format_metrics_txt(
        "Pure 3D U-Net",
        checkpoint,
        data_root,
        tuple(mod.INFER_STEP),
        load_info,
        metrics,
        int(metrics["n_patients"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure U-Net TEST eval with unweighted Dice")
    parser.add_argument("--checkpoint", type=str, help="Path to best_model.pt (or latest_model.pt)")
    parser.add_argument("--data-root", type=str, help="BHSD root with test_images/ and test_masks/")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Results .txt path")
    parser.add_argument("--max-patients", type=int, default=None, help="Limit patients (default: all)")
    parser.add_argument("--merge-batch-size", type=int, default=8)
    args = parser.parse_args()

    default_data = os.environ.get("SEG_DATASET_ROOT")
    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _prompt_path("Checkpoint .pt path")
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else _prompt_path("Dataset root (test_images + test_masks)", default_data)
    )
    if not _test_layout_ok(data_root):
        raise SystemExit(
            f"Expected {data_root}/test_images and test_masks with .nii files. "
            "Set --data-root or SEG_DATASET_ROOT."
        )

    run_eval(
        checkpoint=ckpt,
        data_root=data_root,
        output=Path(args.output).expanduser().resolve(),
        max_patients=args.max_patients,
        merge_batch_size=max(1, args.merge_batch_size),
    )


if __name__ == "__main__":
    main()
