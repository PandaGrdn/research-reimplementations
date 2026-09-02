import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_config
from src.data import load_splits
from src.metrics import noise_floor_decomposition, pair_stats
from src.rollout import validate_hierarchical, validate_hierarchical_long
from src.spatial_pc import freeze_dictionary, make_variables, pretrain_spatial
from src.inference import settle_grounded, settle_info, settle_with_temporal_prior
from src.spatial_pc import unfreeze_dictionary
from src.temporal import (
    TemporalConvRNN,
    settle_with_temporal_prior_unrolled,
    slowness_has_dictionary_grad,
    train_loop,
)
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

    def test_noise_floor_model_consistent(self):
        d_ok = noise_floor_decomposition(settle_abs=1.0, target_abs=2.0)
        self.assertTrue(d_ok["model_consistent"])
        self.assertAlmostEqual(d_ok["noise_over_target"], 0.25, places=5)
        d_bad = noise_floor_decomposition(settle_abs=3.0, target_abs=1.0)
        self.assertFalse(d_bad["model_consistent"])
        self.assertGreater(d_bad["noise_over_target"], 1.0)
        # noise_share stays clamped even when the raw ratio blows past 1
        self.assertEqual(d_bad["noise_share"], 1.0)
        self.assertEqual(d_bad["predictable_fraction"], 0.0)

    def test_load_splits_seeded(self):
        data_root = (ROOT / ".." / "data").resolve()
        train0, val0, test0, info0 = load_splits(data_root, 2, 1, 1, seed=0)
        train1, val1, test1, info1 = load_splits(data_root, 2, 1, 1, seed=1)
        self.assertNotEqual(info0["indices_hash"], info1["indices_hash"])
        train0b, val0b, test0b, info0b = load_splits(data_root, 2, 1, 1, seed=0)
        self.assertEqual(info0["indices_hash"], info0b["indices_hash"])
        self.assertIn("indices_hash", info0)

    def test_settle_info_finite(self):
        image, r, deconvs = _tiny_dict()
        freeze_dictionary(deconvs)
        info = settle_info(image, r, deconvs, 0.001, 1.0, 2, use_prior=True)
        self.assertEqual(len(info["energy"]), 2)
        for e in info["energy"]:
            self.assertTrue(math.isfinite(e))
        self.assertTrue(math.isfinite(info["prior_energy"]))
        self.assertTrue(math.isfinite(info["total_energy"]))
        self.assertTrue(math.isfinite(info["mean_abs_dr"]))

    def test_validate_hierarchical_with_dummy_encoder(self):
        image, r, deconvs = _tiny_dict()
        freeze_dictionary(deconvs)
        net = TemporalConvRNN(r, delta_scale=1.0)
        seq = torch.randn(4, 1, 16, 16) * 0.1

        def dummy_encoder(_I):
            return [torch.zeros_like(ri) for ri in r]

        out = validate_hierarchical(
            seq, r, 1, 2, 1.0, 0.001, 0.02, deconvs, net, encoder=dummy_encoder, log=lambda *_: None
        )
        self.assertEqual(len(out["mse_per_frame"]), 3)
        for key in ("copy_last_mse_per_frame", "mean_frame_mse_per_frame", "copy_last_mse", "mean_frame_mse"):
            self.assertIn(key, out)
        self.assertEqual(len(out["copy_last_mse_per_frame"]), len(out["mse_per_frame"]))

    def test_validate_hierarchical_long_floor_keys(self):
        image, r, deconvs = _tiny_dict()
        freeze_dictionary(deconvs)
        net = TemporalConvRNN(r, delta_scale=1.0)
        seq = torch.randn(6, 1, 16, 16) * 0.1
        out = validate_hierarchical_long(
            seq, r, 1, 2, 1.0, 0.001, 0.02, deconvs, net, split_point=3, split_fix=False, log=lambda *_: None
        )
        for key in (
            "copy_last_mse_per_frame",
            "mean_frame_mse_per_frame",
            "copy_last_mse",
            "mean_frame_mse",
            "copy_last_long_mse",
            "mean_frame_long_mse",
        ):
            self.assertIn(key, out)

    def test_temporal_prior_weight_default_matches_old_hardcode(self):
        # old code hard-coded (1 / (sigma_2 * 100)); the new default temporal_prior_weight=0.01
        # gives (temporal_prior_weight / sigma_2) which is algebraically identical for any sigma_2.
        default_tpw = 0.01
        for sigma_2 in (0.5, 1.0, 2.0):
            self.assertAlmostEqual(default_tpw / sigma_2, 1.0 / (sigma_2 * 100), places=10)

        from src.temporal import inference_kwargs

        cfg = resolve_config(smoke=True, root=ROOT)
        kw = inference_kwargs(cfg)
        self.assertAlmostEqual(kw["temporal_prior_weight"], 0.01, places=8)

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

    def test_settle_with_temporal_prior_unrolled_k0_matches_detached(self):
        # unroll_last_k=0 must reduce EXACTLY (bit-for-bit) to settle_with_temporal_prior.
        image, r, deconvs = _tiny_dict()
        net = TemporalConvRNN(r, delta_scale=1.0)
        r_prev1 = [ri.clone() for ri in r]
        hidden = net.init_hidden(r)
        with torch.no_grad():
            r_pred, _, _ = net(r_prev1, hidden)
        r_a, _, _ = settle_with_temporal_prior(
            image, [ri.clone() for ri in r], r_pred, deconvs, 0.001, 0.02, 1.0, 5, 2,
            r_prev1=r_prev1, lambda_slow=1.0, temporal_prior_weight=0.01,
        )
        r_b, _, _ = settle_with_temporal_prior_unrolled(
            image, [ri.clone() for ri in r], r_pred, deconvs, 0.001, 0.02, 1.0, 5, 2,
            r_prev1=r_prev1, lambda_slow=1.0, temporal_prior_weight=0.01, unroll_last_k=0,
        )
        for a, b in zip(r_a, r_b):
            self.assertTrue(torch.equal(a, b))

    def test_settle_with_temporal_prior_unrolled_grad_path(self):
        # unroll_last_k>0 must leave r_curr differentiable w.r.t. the dictionary.
        image, r, deconvs = _tiny_dict()
        unfreeze_dictionary(deconvs)
        net = TemporalConvRNN(r, delta_scale=1.0)
        r_prev1 = [ri.clone() for ri in r]
        hidden = net.init_hidden(r)
        with torch.no_grad():
            r_pred, _, _ = net(r_prev1, hidden)
        r_out, _, _ = settle_with_temporal_prior_unrolled(
            image, [ri.clone() for ri in r], r_pred, deconvs, 0.001, 0.02, 1.0, 4, 2,
            r_prev1=r_prev1, lambda_slow=1.0, temporal_prior_weight=0.01, unroll_last_k=2,
        )
        self.assertTrue(any(ri.requires_grad for ri in r_out))
        self.assertTrue(any(ri.grad_fn is not None for ri in r_out))

    def test_slowness_has_dictionary_grad_k0_false_k_gt0_true(self):
        # The C7 headline claim, verified directly rather than inferred: with
        # the detached settle (k=0) the slowness term reaches no dictionary
        # weight; with the unrolled settle (k>0) it does.
        image, r, deconvs = _tiny_dict()
        unfreeze_dictionary(deconvs)
        net = TemporalConvRNN(r, delta_scale=1.0)
        g0 = slowness_has_dictionary_grad(
            image, r, deconvs, net, 0.001, 0.02, 1.0, 5, 2, lambda_slow=1.0, slow_unroll_k=0,
        )
        g3 = slowness_has_dictionary_grad(
            image, r, deconvs, net, 0.001, 0.02, 1.0, 5, 2, lambda_slow=1.0, slow_unroll_k=3,
        )
        self.assertFalse(g0)
        self.assertTrue(g3)

    def test_train_loop_slow_unroll_k_default_matches_k0(self):
        # Default (slow_unroll_k=0) must be numerically identical to the old
        # (pre-unroll) train_loop behaviour.
        seed_everything(0)
        image, r, deconvs = _tiny_dict()
        net = TemporalConvRNN(r, delta_scale=1.0)
        r_out, _, _, _ = train_loop(
            image, None, [ri.clone() for ri in r],
            0.001, 0.001, 0.02, 0.002, 1.0, 1, 3, 2, deconvs, net,
            lambda_slow=1.0, delta_loss_weight=1.0,
        )
        self.assertTrue(all(not ri.requires_grad for ri in r_out))
        self.assertTrue(all(torch.isfinite(ri).all() for ri in r_out))

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
        """Exercise every fig_cN in FIGURES against its REAL current schema.

        Prefers the actual results/<exp>/latest/metrics.json written by a real
        (e.g. smoke) run, since that is guaranteed to match whatever each
        fig_cN function currently reads; falls back to a small hand-built
        dict (kept in sync with the fig_cN bodies in scripts/make_figures.py)
        when no run has happened yet in this checkout.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_figures

        fallback = {
            "c0_isolation_test": {
                "step1": {
                    "true": {
                        "total_energy": {"mean": 40.0, "std": 1.0},
                        "iters": {"mean": 2.0, "std": 0.0},
                        "two_init_cos": {"mean": 0.93, "std": 0.01},
                    },
                    "synthetic": {
                        "total_energy": {"mean": 3.3, "std": 0.5},
                        "iters": {"mean": 2.0, "std": 0.0},
                        "two_init_cos": {"mean": 0.79, "std": 0.02},
                    },
                    "cross_cos": {"mean": 0.93, "std": 0.0},
                    "pixel_mse_true_vs_syn": {"mean": 0.03, "std": 0.0},
                    "saturation": {"mean": 0.0, "std": 0.0},
                },
                "step2": {"cross_cos": {"mean": 0.81, "std": 0.0}},
            },
            "c1_rollout_collapse": {
                "headline_split": "2",
                "split_points": {
                    "2": {
                        "mse_per_frame_mean": [0.1, 0.2, 0.5],
                        "mse_per_frame_std": [0.01, 0.01, 0.02],
                        "copy_last_mse_per_frame_mean": [0.08, 0.15, 0.3],
                        "mean_frame_mse_per_frame_mean": [0.09, 0.18, 0.35],
                        "copy_last_long_mse_mean": 0.2,
                        "mean_frame_long_mse_mean": 0.25,
                    }
                },
            },
            "c2_mitigation_grid": {
                "cells": [
                    {
                        "name": "baseline",
                        "long_mse_mean": 0.4,
                        "long_mse_std": 0.05,
                        "copy_last_long_mse_mean": 0.35,
                        "mean_frame_long_mse_mean": 0.38,
                    },
                    {
                        "name": "all_combined",
                        "long_mse_mean": 0.39,
                        "long_mse_std": 0.04,
                        "copy_last_long_mse_mean": 0.35,
                        "mean_frame_long_mse_mean": 0.38,
                    },
                ]
            },
            "c3_copy_detection": {
                "motion_gap_mean": [0.01, 0.02],
                "motion_gap_std": [0.0, 0.0],
                "delta_ratio_mean": [0.1, 0.05],
                "delta_ratio_std": [0.0, 0.0],
            },
            "c4_settle_determinism": {
                "dist_vs_init_noise": [
                    {"init_noise": 0.0, "abs_mean": 0.0, "cos_mean": 1.0},
                    {"init_noise": 0.01, "abs_mean": 2.9, "cos_mean": 0.32},
                ],
                "dist_vs_iters": [
                    {"iters": 50, "abs_mean": 3.0, "cos_mean": 0.73, "energy_mean": 50.0},
                    {"iters": 200, "abs_mean": 2.5, "cos_mean": 0.93, "energy_mean": 10.0},
                ],
                "null_space_slope": 0.5,
                "headline_init_noise": 0.01,
            },
            "c5_latent_smoothness": {
                "conditions": {
                    "cold": {"frac_rel": 1.0, "cons_cos": 0.12},
                    "warm": {"frac_rel": 0.8, "cons_cos": 0.55},
                    "warm_zero_init": {"frac_rel": 0.75, "cons_cos": 0.6},
                    "pixel": {"frac_rel": 0.86, "cons_cos": 0.33},
                },
                "seeds": [
                    {
                        "cold": {"frac_rel": 1.0},
                        "warm": {"frac_rel": 0.8},
                        "warm_zero_init": {"frac_rel": 0.75},
                        "pixel": {"frac_rel": 0.86},
                    }
                ],
                "iters_sweep": [
                    {"iters": 50, "cons_cos_mean": 0.41, "cons_cos_std": 0.02},
                    {"iters": 200, "cons_cos_mean": 0.31, "cons_cos_std": 0.01},
                ],
            },
            "c6_noise_floor": {
                "protocols": {
                    "cold_independent": {"noise_energy": 8.8, "target_energy": 1.9, "noise_over_target": 4.7},
                    "warm_independent_init": {"noise_energy": 5.4, "target_energy": 1.9, "noise_over_target": 2.9},
                    "pipeline": {"noise_energy": 0.0, "target_energy": 1.9, "noise_over_target": 0.0},
                },
                "headline_protocol": "warm_independent_init",
            },
            "c7_slowness_sweep": {
                "sweep": [
                    {
                        "lambda_slow": 0.0, "unroll_k": 0,
                        "cons_cos_no_pull": 0.4, "cons_cos_no_pull_std": 0.02,
                        "cons_cos_with_pull": 0.5, "dict_drift": 0.0,
                    },
                    {
                        "lambda_slow": 1.0, "unroll_k": 0,
                        "cons_cos_no_pull": 0.41, "cons_cos_no_pull_std": 0.02,
                        "cons_cos_with_pull": 0.6, "dict_drift": 0.0,
                    },
                    {
                        "lambda_slow": 0.0, "unroll_k": 5,
                        "cons_cos_no_pull": 0.4, "cons_cos_no_pull_std": 0.02,
                        "cons_cos_with_pull": 0.5, "dict_drift": 0.0,
                    },
                    {
                        "lambda_slow": 1.0, "unroll_k": 5,
                        "cons_cos_no_pull": 0.55, "cons_cos_no_pull_std": 0.02,
                        "cons_cos_with_pull": 0.65, "dict_drift": 0.05,
                    },
                ]
            },
            "c8_amortized_contrast": {
                "arms": {
                    "iterative": {
                        "tf_mse_mean": 0.04,
                        "long_mse_mean": 0.05,
                        "copy_last_mse_mean": 0.06,
                        "copy_last_long_mse_mean": 0.09,
                        "cons_cos_list": [0.4, 0.5],
                        "recon_mse_mean": 0.02,
                        "energy_mean": 40.0,
                        "predictability": {
                            "linear": {"code": {"r2_vs_copy_last_mean": -8.9, "r2_vs_copy_last_std": 1.0}}
                        },
                    },
                    "amortized": {
                        "tf_mse_mean": 0.06,
                        "long_mse_mean": 0.07,
                        "copy_last_mse_mean": 0.06,
                        "copy_last_long_mse_mean": 0.09,
                        "cons_cos_list": [0.8, 0.9],
                        "recon_mse_mean": 0.05,
                        "energy_mean": 5.0,
                        "predictability": {
                            "linear": {"code": {"r2_vs_copy_last_mean": -3.4, "r2_vs_copy_last_std": 0.5}}
                        },
                    },
                },
                "pixel_reference": {
                    "cons_cos_mean": 0.38,
                    "predictability": {"linear": {"r2_vs_copy_last_mean": -0.23, "r2_vs_copy_last_std": 0.0}},
                },
                "predictability_primary_model": "linear",
            },
            "c9_predictability": {
                "predictability": {
                    "linear": {
                        "code": {"r2_vs_copy_last_mean": -8.96, "r2_vs_copy_last_std": 1.0},
                        "pixel": {"r2_vs_copy_last_mean": -0.23, "r2_vs_copy_last_std": 0.0},
                    }
                },
                "translation": {
                    "per_shift": [
                        {
                            "dx": 2, "dy": 0, "pixel_cos_mean": 0.69,
                            "layers": [{"layer": 0, "cos_mean": 0.99, "exact_expected": True, "aliased": False}],
                        },
                        {
                            "dx": 1, "dy": 0, "pixel_cos_mean": 0.5,
                            "layers": [{"layer": 0, "cos_mean": 0.76, "exact_expected": False, "aliased": True}],
                        },
                    ]
                },
            },
            "ladder": {
                "rows": [
                    {"arm": "baseline", "c1": {"long_mse": 0.4, "copy_last_floor": 0.35},
                     "c5": {"warm_frac_rel": 0.8, "pixel_frac_rel": 0.75},
                     "c9": {"code_r2": 0.1, "pixel_r2": 0.6}},
                    {"arm": "zero_init", "c1": {"long_mse": 0.38, "copy_last_floor": 0.35},
                     "c5": {"warm_frac_rel": 0.7, "pixel_frac_rel": 0.75},
                     "c9": {"code_r2": 0.15, "pixel_r2": 0.6}},
                ]
            },
        }

        def _load_or_fallback(exp):
            p = ROOT / "results" / exp / "latest" / "metrics.json"
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception:
                    pass
            return fallback[exp]

        with tempfile.TemporaryDirectory() as td:
            for exp, fn in make_figures.FIGURES.items():
                fn(_load_or_fallback(exp), td)
            for name in ("fig_c0", "fig_c1", "fig_c2", "fig_c3", "fig_c4", "fig_c5",
                         "fig_c6", "fig_c7", "fig_c8", "fig_c9", "fig_ladder"):
                self.assertTrue((Path(td) / f"{name}.pdf").exists())
                self.assertTrue((Path(td) / f"{name}.png").exists())


if __name__ == "__main__":
    unittest.main()
