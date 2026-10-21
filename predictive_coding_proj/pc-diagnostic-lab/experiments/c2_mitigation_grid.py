#!/usr/bin/env python3
"""C2 — standard mitigations do not fix long-horizon collapse."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    eval_long_rollouts,
    load_data,
    overlay_temporal,
    parse_args,
    setup,
    train_temporal_pc,
)
from src.utils import finish_run, mean_std, new_run_dir, seed_everything

CELLS = [
    {"name": "baseline", "ss_p": 0.0, "ss_p_end": None, "rollout_k": 1, "r_noise_std": 0.0, "r_noise_std_end": None, "delta_bounded": False, "split_fix": False},
    {"name": "scheduled_sampling", "ss_p": 0.4, "ss_p_end": None, "rollout_k": 1, "r_noise_std": 0.0, "r_noise_std_end": None, "delta_bounded": False, "split_fix": False},
    {"name": "rollout_loss", "ss_p": 0.0, "ss_p_end": None, "rollout_k": 4, "r_noise_std": 0.0, "r_noise_std_end": None, "delta_bounded": False, "split_fix": False},
    {"name": "r_noise", "ss_p": 0.0, "ss_p_end": None, "rollout_k": 1, "r_noise_std": 0.05, "r_noise_std_end": None, "delta_bounded": False, "split_fix": False},
    {"name": "delta_bounded", "ss_p": 0.0, "ss_p_end": None, "rollout_k": 1, "r_noise_std": 0.0, "r_noise_std_end": None, "delta_bounded": True, "split_fix": False},
    {"name": "split_fix", "ss_p": 0.0, "ss_p_end": None, "rollout_k": 1, "r_noise_std": 0.0, "r_noise_std_end": None, "delta_bounded": False, "split_fix": True},
    {"name": "all_combined", "ss_p": 0.4, "ss_p_end": None, "rollout_k": 4, "r_noise_std": 0.05, "r_noise_std_end": None, "delta_bounded": True, "split_fix": True},
    # These two are inference-config overrides (temporal_prior_weight lives under
    # cfg["inference"], not cfg["temporal"]), so they carry an "inference_overlay"
    # dict instead of the flat temporal-key fields the cells above use.
    {"name": "temporal_prior_off", "inference_overlay": {"temporal_prior_weight": 0.0}},
    {"name": "temporal_prior_strong", "inference_overlay": {"temporal_prior_weight": 1.0}},
    # Follow-ups, not in the default grid: in the cells above the slowness pull
    # (temporal.lambda_slow=3) points the same way as the temporal prior and is
    # 3-300x stronger, so the prior's weight barely moves the settled code.
    # These remove the slowness pull, with the prior at its default and at 1.0.
    {"name": "slowness_off", "temporal_overlay": {"lambda_slow": 0.0}},
    {"name": "prior_strong_slowness_off", "temporal_overlay": {"lambda_slow": 0.0},
     "inference_overlay": {"temporal_prior_weight": 1.0}},
]
DEFAULT_CELLS = [c["name"] for c in CELLS[:9]]


def overlay_inference(cfg, **kwargs):
    out = copy.deepcopy(cfg)
    out["inference"].update(kwargs)
    return out


def selected_cells(cfg):
    names = cfg.get("c2", {}).get("cells") or DEFAULT_CELLS
    wanted = set(names)
    return [c for c in CELLS if c["name"] in wanted]


def build_cell_cfg(cfg, cell):
    if "inference_overlay" in cell or "temporal_overlay" in cell:
        out = cfg
        if "temporal_overlay" in cell:
            out = overlay_temporal(out, **cell["temporal_overlay"])
        if "inference_overlay" in cell:
            out = overlay_inference(out, **cell["inference_overlay"])
        return out
    return overlay_temporal(cfg, **{k: v for k, v in cell.items() if k != "name"})


def summarize(results, headline_sp):
    """Build the C2 metrics dict from per-cell records (also used by run_parallel's merge)."""
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    floor_ref = baseline or (results[0] if results else None)
    summary = "C2 | " + "  ".join(f"{r['name']}={r['long_mse_mean']:.4f}" for r in results)
    if baseline is not None:
        summary += f"  (baseline {baseline['long_mse_mean']:.4f})"
    if floor_ref is not None:
        summary += (
            f"  | copy-last floor {floor_ref['copy_last_long_mse_mean']:.4f} "
            f"| mean-frame floor {floor_ref['mean_frame_long_mse_mean']:.4f}"
        )
        if floor_ref.get("blank_long_mse_mean") is not None:
            summary += f" | blank floor {floor_ref['blank_long_mse_mean']:.4f}"
    return {"claim": "C2", "headline_split": headline_sp, "cells": results, "summary": summary}


def main():
    args = parse_args("C2 mitigation grid")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c2_mitigation_grid", arm=arm_name(cfg))
    headline_sp = cfg["eval"].get("headline_split", 10)
    cells = selected_cells(cfg)
    results = []

    for cell in cells:
        long_vals, cl_floor_vals, mf_floor_vals, blank_vals, mass_vals, corr_vals = [], [], [], [], [], []
        for seed in seeds:
            seed_everything(seed)
            cell_cfg = build_cell_cfg(cfg, cell)
            cell_cfg["seed"] = seed
            train, val, test, info = load_data(cell_cfg, root, seed)
            print(f"cell={cell['name']} seed={seed}  {info['n_train']} train  hash={info['split_hash']}")
            deconvs, r_init, _ = ensure_dictionary(cell_cfg, train, device, root)
            model = build_temporal(cell_cfg, r_init, device)
            model, _, _ = train_temporal_pc(
                train, r_init, deconvs, model, cell_cfg, device, val_seq=val[0], log=print
            )
            long = eval_long_rollouts(
                test, r_init, deconvs, model, cell_cfg, device, split_points=[headline_sp], log=print
            )
            payload = long[str(headline_sp)]
            long_vals.append(payload["long_mse_mean"])
            cl_floor_vals.append(payload["copy_last_long_mse_mean"])
            mf_floor_vals.append(payload["mean_frame_long_mse_mean"])
            blank_vals.append(payload["blank_long_mse_mean"])
            mass_vals.append(payload["mass_ratio_long_mean"])
            corr_vals.append(payload["corr_long_mean"])
        mu, sd = mean_std(long_vals)
        cl_mu, cl_sd = mean_std(cl_floor_vals)
        mf_mu, mf_sd = mean_std(mf_floor_vals)
        rec = {
            "name": cell["name"],
            **{k: v for k, v in cell.items() if k != "name"},
            "long_mse_mean": mu,
            "long_mse_std": sd,
            "copy_last_long_mse_mean": cl_mu,
            "copy_last_long_mse_std": cl_sd,
            "mean_frame_long_mse_mean": mf_mu,
            "mean_frame_long_mse_std": mf_sd,
            "per_seed": long_vals,
            "seeds": list(seeds),
            "blank_long_mse_mean": mean_std(blank_vals)[0],
            "mass_ratio_long_mean": mean_std(mass_vals)[0],
            "per_seed_mass_ratio": mass_vals,
            "per_seed_corr": corr_vals,
            "corr_long_mean": mean_std(corr_vals)[0],
        }
        results.append(rec)
        print(f"C2 | {cell['name']}: {mu:.4f} ± {sd:.4f}  (copy-last floor {cl_mu:.4f})")

    metrics = summarize(results, headline_sp)
    summary = metrics["summary"]
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
