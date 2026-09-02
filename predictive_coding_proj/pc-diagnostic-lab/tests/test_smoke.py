import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_config
from src.metrics import noise_floor_decomposition, pair_stats
from src.rollout import validate_hierarchical_long
from src.spatial_pc import freeze_dictionary, make_variables, pretrain_spatial
from src.inference import settle_grounded
from src.temporal import TemporalConvRNN, train_loop
from src.utils import seed_everything


def _tiny_dict(device="cpu"):
    image = torch.randn(1, 1, 16, 16, device=device) * 0.1
    _, r, deconvs = make_variables(image, initial_r_size=16, num_layers=2, device=device)
    return image, r, deconvs


class CoreTests(unittest.TestCase):
    def setUp(self):
        seed_everything(0)
        self.device = torch.device("cpu")

    def test_pair_stats_identity(self):
        r = [torch.randn(1, 4, 2, 2)]
        s = pair_stats(r, r)
        self.assertAlmostEqual(s["cos"], 1.0, places=5)
        self.assertAlmostEqual(s["abs"], 0.0, places=5)

    def test_noise_floor_formula(self):
        # settle_abs = σ√2 ⇒ 2σ² = settle_abs². If target = settle_abs, noise share = 1.
        d = noise_floor_decomposition(settle_abs=2.0, target_abs=2.0)
        self.assertAlmostEqual(d["noise_share"], 1.0, places=5)
        self.assertAlmostEqual(d["predictable_fraction"], 0.0, places=5)
        d2 = noise_floor_decomposition(settle_abs=1.0, target_abs=2.0)
        self.assertAlmostEqual(d2["noise_share"], 0.25, places=5)
        self.assertAlmostEqual(d2["predictable_fraction"], 0.75, places=5)

    def test_make_variables_shapes(self):
        image, r, deconvs = _tiny_dict()
        self.assertEqual(len(r), 2)
        self.assertEqual(tuple(r[0].shape), (1, 32, 8, 8))
        self.assertEqual(tuple(r[1].shape), (1, 64, 4, 4))
        self.assertEqual(len(deconvs), 2)

    def test_settle_grounded_runs(self):
        image, r, deconvs = _tiny_dict()
        freeze_dictionary(deconvs)
        r1, n1 = settle_grounded(image, r, deconvs, 0.001, 0.02, 1.0, 2, 2)
        r2, n2 = settle_grounded(image, r, deconvs, 0.001, 0.02, 1.0, 2, 2)
        self.assertEqual(n1, 2)
        s = pair_stats(r1, r2)
        self.assertTrue(-1.0 <= s["cos"] <= 1.0)

    def test_temporal_zero_readout_is_identity(self):
        image, r, deconvs = _tiny_dict()
        net = TemporalConvRNN(r, delta_scale=1.0, delta_bounded=True)
        pred, hidden, deltas = net(r, None)
        for d in deltas:
            self.assertTrue(torch.allclose(d, torch.zeros_like(d), atol=1e-6))
        for a, b in zip(pred, r):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_train_loop_one_step(self):
        image, r, deconvs = _tiny_dict()
        net = TemporalConvRNN(r, delta_scale=1.0)
        r_out, r_pred, hidden, stats = train_loop(
            image, None, [ri.clone() for ri in r],
            0.001, 0.001, 0.02, 0.002, 1.0, 1, 2, 2, deconvs, net,
            lambda_slow=0.0, delta_loss_weight=1.0,
        )
        self.assertEqual(len(r_out), 2)
        self.assertEqual(len(hidden), 2)
        self.assertTrue(all(torch.isfinite(ri).all() for ri in r_pred))

    def test_split_fix_flag_changes_wiring(self):
        image, r, deconvs = _tiny_dict()
        freeze_dictionary(deconvs)
        net = TemporalConvRNN(r, delta_scale=1.0)
        seq = torch.randn(6, 1, 16, 16) * 0.1
        a = validate_hierarchical_long(
            seq, r, 1, 2, 1.0, 0.001, 0.02, deconvs, net, split_point=3, split_fix=False, log=lambda *_: None
        )
        b = validate_hierarchical_long(
            seq, r, 1, 2, 1.0, 0.001, 0.02, deconvs, net, split_point=3, split_fix=True, log=lambda *_: None
        )
        self.assertEqual(len(a["mse_per_frame"]), 5)
        self.assertFalse(a["split_fix"])
        self.assertTrue(b["split_fix"])
        self.assertIn("long_mse", a)

    def test_pretrain_one_epoch(self):
        image, r, deconvs = _tiny_dict()
        frames = [torch.randn(1, 16, 16) * 0.1 for _ in range(3)]
        deconvs, mse, vis = pretrain_spatial(
            frames, r, deconvs, 0.001, 0.001, 0.02, 0.002, 1.0, 1, 2, 2, log=lambda *_: None
        )
        self.assertTrue(mse >= 0.0)
        self.assertIsNotNone(vis)

    def test_smoke_config_tiny(self):
        cfg = resolve_config(smoke=True, root=ROOT)
        self.assertEqual(cfg["data"]["n_train"], 2)
        self.assertEqual(cfg["inference"]["num_epochs_inner"], 2)
        self.assertEqual(cfg["temporal"]["epochs"], 1)
        self.assertIn("baseline", cfg["c2"]["cells"])

    def test_make_figures_from_dummy(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_figures

        dummy = {
            "c1_rollout_collapse": {
                "split_points": {
                    "2": {"mse_per_frame_mean": [0.1, 0.2, 0.5], "mse_per_frame_std": [0.01, 0.01, 0.02]}
                }
            },
            "c2_mitigation_grid": {
                "cells": [
                    {"name": "baseline", "long_mse_mean": 0.4, "long_mse_std": 0.05},
                    {"name": "all_combined", "long_mse_mean": 0.39, "long_mse_std": 0.04},
                ]
            },
            "c3_copy_detection": {
                "motion_gap_mean": [0.01, 0.02],
                "motion_gap_std": [0.0, 0.0],
                "delta_ratio_mean": [0.1, 0.05],
                "delta_ratio_std": [0.0, 0.0],
            },
            "c4_settle_determinism": {
                "cos_vs_iters": [
                    {"iters": 50, "cos_mean": 0.73, "cos_std": 0.02},
                    {"iters": 200, "cos_mean": 0.93, "cos_std": 0.01},
                ]
            },
            "c5_latent_smoothness": {
                "cold": {"cons_cos_list": [0.4, 0.5], "un_cos_list": [0.3, 0.35], "frac_rel_mean": 0.9},
                "warm": {"cons_cos_list": [0.55, 0.6], "un_cos_list": [0.3, 0.32], "frac_rel_mean": 0.8},
            },
            "c6_noise_floor": {"noise_share_mean": 0.7, "predictable_fraction_mean": 0.3},
            "c7_slowness_sweep": {
                "sweep": [
                    {"lambda_slow": 0.0, "cons_cos_mean": 0.4, "cons_cos_std": 0.02},
                    {"lambda_slow": 1.0, "cons_cos_mean": 0.45, "cons_cos_std": 0.02},
                ]
            },
            "c8_amortized_contrast": {
                "iterative": {"cons_cos_list": [0.4], "un_cos_list": [0.3]},
                "amortized": {"cons_cos_list": [0.8], "un_cos_list": [0.3], "same_frame_cos_mean": 1.0},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            for exp, fn in make_figures.FIGURES.items():
                fn(dummy[exp], td)
            self.assertTrue((Path(td) / "fig_c1.pdf").exists())
            self.assertTrue((Path(td) / "fig_c6.png").exists())


if __name__ == "__main__":
    unittest.main()
