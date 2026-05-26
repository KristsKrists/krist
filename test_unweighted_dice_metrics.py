# -*- coding: utf-8 -*-
"""
Teaching template: smoke tests for both segmentation model templates.

Validates `mamba3d_unet_template.py` (Mamba3DUNet) and `pure3d_unet_template.py` (UNet3D):
- Forward pass: output shape, finite logits
- Architecture: 10 Mamba blocks in the configured layout; pure U-Net has no Mamba
- Metrics: unweighted soft/hard macro foreground Dice (shared helpers)
- Loss: weighted combined loss runs on dummy batch
- Checkpoint: state_dict save/load roundtrip (optional smoke test)

No dataset or GPU required. Uses small spatial sizes for fast CPU runs.

Run from repo root:
  python github_templates/test_unweighted_dice_metrics.py

From this directory:
  cd github_templates && python test_unweighted_dice_metrics.py

Run one test class:
  python test_unweighted_dice_metrics.py TestModelForwardBoth -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn

# =============================================================================
# === USER CONFIG (align with training templates) ==============================
# =============================================================================

IN_CHANNELS = 4
NUM_CLASSES = 6
BASE_CHANNELS = 16
SSM_DIM = 16
DROPOUT = 0.1

# Small patch for smoke tests (N, C, D, H, W). Must survive 3× MaxPool3d(2) and enc4 pool (1,2,2)
# so bottleneck keeps >1 spatial voxels (InstanceNorm in train mode). 8×32×32 is a practical minimum.
TEST_BATCH = 1
TEST_D = 8
TEST_H = 32
TEST_W = 32

MAMBA_TEMPLATE_FILE = "mamba3d_unet_template.py"
PURE_TEMPLATE_FILE = "pure3d_unet_template.py"

MAMBA_BLOCK_ATTRS = (
    "enc1_mamba_1",
    "enc2_mamba_1",
    "enc2_mamba_2",
    "enc3_mamba_1",
    "enc3_mamba_2",
    "dec3_mamba_1",
    "dec3_mamba_2",
    "dec2_mamba_1",
    "dec2_mamba_2",
    "dec1_mamba_1",
)

LEGACY_MAMBA_ATTRS = (
    "enc2_m",
    "enc4_m1",
    "enc4_m2",
    "bot_m1",
    "bot_m2",
    "dec4_m1",
    "dec4_m2",
)


def _load_template_module(module_name: str, filename: str) -> Any:
    here = Path(__file__).resolve().parent
    path = here / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dummy_input(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return torch.randn(TEST_BATCH, IN_CHANNELS, TEST_D, TEST_H, TEST_W, device=device)


def _dummy_one_hot(fg_class: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    y = torch.zeros(TEST_BATCH, NUM_CLASSES, TEST_D, TEST_H, TEST_W, device=device)
    y[:, fg_class] = 1.0
    return y


def _build_mamba(mod: Any, device: torch.device) -> nn.Module:
    return mod.Mamba3DUNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        ssm_dim=SSM_DIM,
        dropout=DROPOUT,
    ).to(device)


def _build_pure(mod: Any, device: torch.device) -> nn.Module:
    return mod.UNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT,
    ).to(device)


class _TemplateFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mamba_mod = _load_template_module("_mamba_tpl_test", MAMBA_TEMPLATE_FILE)
        cls.pure_mod = _load_template_module("_pure_tpl_test", PURE_TEMPLATE_FILE)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _models(self) -> List[Tuple[str, nn.Module]]:
        return [
            ("Mamba3DUNet", _build_mamba(self.mamba_mod, self.device)),
            ("UNet3D", _build_pure(self.pure_mod, self.device)),
        ]


class TestModelForwardBoth(_TemplateFixtures):
    def test_output_shape_and_finite_logits(self) -> None:
        x = _dummy_input(self.device)
        expected = (TEST_BATCH, NUM_CLASSES, TEST_D, TEST_H, TEST_W)
        for name, model in self._models():
            model.eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(tuple(out.shape), expected, msg=name)
            self.assertTrue(torch.isfinite(out).all().item(), msg=f"{name} logits must be finite")
            self.assertFalse(torch.allclose(out, torch.zeros_like(out)), msg=f"{name} logits all zero")

    def _assert_one_backward_step(self, name: str, model: nn.Module, mod: Any) -> None:
        x = _dummy_input(self.device)
        y = _dummy_one_hot(1, self.device)
        cw = torch.ones(NUM_CLASSES, device=self.device)
        model.train()
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        loss, _, _ = mod.combined_loss_components_probs(y, probs, cw)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        self.assertTrue(has_grad, msg=f"{name} should receive gradients")

    def test_pure_train_backward_one_step(self) -> None:
        self._assert_one_backward_step("UNet3D", _build_pure(self.pure_mod, self.device), self.pure_mod)

    @unittest.skipUnless(torch.cuda.is_available(), "Mamba backward smoke test needs CUDA (high CPU memory)")
    def test_mamba_train_backward_one_step(self) -> None:
        self._assert_one_backward_step("Mamba3DUNet", _build_mamba(self.mamba_mod, self.device), self.mamba_mod)


class TestMambaArchitecture(_TemplateFixtures):
    def test_active_mamba_blocks(self) -> None:
        model = _build_mamba(self.mamba_mod, torch.device("cpu"))
        for attr in MAMBA_BLOCK_ATTRS:
            self.assertTrue(hasattr(model, attr), msg=f"missing {attr}")
        for attr in LEGACY_MAMBA_ATTRS:
            self.assertFalse(hasattr(model, attr), msg=f"legacy block still present: {attr}")

    def test_bottleneck_conv_no_mamba_in_forward_graph(self) -> None:
        model = _build_mamba(self.mamba_mod, torch.device("cpu"))
        self.assertTrue(hasattr(model, "bottleneck_conv"))
        self.assertFalse(hasattr(model, "bot_conv"))


class TestPureArchitecture(_TemplateFixtures):
    def test_no_mamba_modules(self) -> None:
        model = _build_pure(self.pure_mod, torch.device("cpu"))
        mamba_names = [n for n, _ in model.named_modules() if "mamba" in n.lower()]
        self.assertEqual(mamba_names, [], msg=f"unexpected Mamba modules: {mamba_names}")


class TestUnweightedDiceMetrics(_TemplateFixtures):
    def test_perfect_prediction_soft_and_hard_near_one(self) -> None:
        y, p = _dummy_one_hot(1), _dummy_one_hot(1)
        for mod in (self.mamba_mod, self.pure_mod):
            sd = mod.unweighted_soft_dice_macro_fg(y, p)
            hd = mod.unweighted_hard_dice_macro_fg(y, p)
            self.assertGreater(sd, 0.999, msg=f"{mod.__name__} soft dice perfect")
            self.assertGreater(hd, 0.999, msg=f"{mod.__name__} hard dice perfect")

    def test_mamba_and_pure_unweighted_metrics_match(self) -> None:
        torch.manual_seed(0)
        y = _dummy_one_hot(1)
        p = torch.softmax(torch.randn_like(y), dim=1)
        sm = self.mamba_mod.unweighted_soft_dice_macro_fg(y, p)
        sp = self.pure_mod.unweighted_soft_dice_macro_fg(y, p)
        hm = self.mamba_mod.unweighted_hard_dice_macro_fg(y, p)
        hp = self.pure_mod.unweighted_hard_dice_macro_fg(y, p)
        self.assertAlmostEqual(sm, sp, places=6)
        self.assertAlmostEqual(hm, hp, places=6)

    def test_wrong_class_lower_than_perfect_soft(self) -> None:
        y, perfect = _dummy_one_hot(1), _dummy_one_hot(1)
        wrong = torch.zeros_like(y)
        wrong[:, 2] = 1.0
        sm_good = self.mamba_mod.unweighted_soft_dice_macro_fg(y, perfect)
        sm_bad = self.mamba_mod.unweighted_soft_dice_macro_fg(y, wrong)
        self.assertGreater(sm_good, sm_bad + 0.05)

    def test_hard_dice_argmax_sensitivity(self) -> None:
        y = _dummy_one_hot(1)
        logits_right = torch.zeros_like(y)
        logits_right[:, 1] = 10.0
        p_right = torch.softmax(logits_right, dim=1)
        logits_wrong = torch.zeros_like(y)
        logits_wrong[:, 2] = 10.0
        p_wrong = torch.softmax(logits_wrong, dim=1)
        hr = self.pure_mod.unweighted_hard_dice_macro_fg(y, p_right)
        hw = self.pure_mod.unweighted_hard_dice_macro_fg(y, p_wrong)
        self.assertGreater(hr, hw + 0.05)

    def test_single_channel_returns_zero(self) -> None:
        y = torch.zeros(1, 1, 2, 2, 2)
        p = torch.zeros(1, 1, 2, 2, 2)
        self.assertEqual(self.mamba_mod.unweighted_soft_dice_macro_fg(y, p), 0.0)
        self.assertEqual(self.mamba_mod.unweighted_hard_dice_macro_fg(y, p), 0.0)


class TestCombinedLossBoth(_TemplateFixtures):
    def test_loss_finite_on_random_batch(self) -> None:
        y = _dummy_one_hot(2, self.device)
        cw = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0], device=self.device)
        for name, model in self._models():
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(_dummy_input(self.device)), dim=1)
            mod = self.mamba_mod if name == "Mamba3DUNet" else self.pure_mod
            loss, gdl, wce = mod.combined_loss_components_probs(y, probs, cw)
            self.assertTrue(torch.isfinite(loss).item(), msg=name)
            self.assertGreaterEqual(gdl, 0.0)
            self.assertGreaterEqual(wce, 0.0)


class TestCheckpointRoundtrip(_TemplateFixtures):
    def _roundtrip(self, build_fn: Callable[[], nn.Module], label: str) -> None:
        x = _dummy_input(self.device)
        model = build_fn()
        model.eval()
        with torch.no_grad():
            ref = model(x).cpu()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{label}.pt"
            torch.save({"model_state_dict": model.state_dict()}, path)
            model2 = build_fn()
            try:
                ck = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                ck = torch.load(path, map_location=self.device)
            model2.load_state_dict(ck["model_state_dict"])
            model2.eval()
            with torch.no_grad():
                out = model2(x).cpu()
        self.assertTrue(torch.allclose(ref, out, atol=1e-5, rtol=1e-4), msg=label)

    def test_mamba_checkpoint_roundtrip(self) -> None:
        self._roundtrip(lambda: _build_mamba(self.mamba_mod, self.device), "mamba")

    def test_pure_checkpoint_roundtrip(self) -> None:
        self._roundtrip(lambda: _build_pure(self.pure_mod, self.device), "pure")


def run_all() -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestModelForwardBoth,
        TestMambaArchitecture,
        TestPureArchitecture,
        TestUnweightedDiceMetrics,
        TestCombinedLossBoth,
        TestCheckpointRoundtrip,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_all()
