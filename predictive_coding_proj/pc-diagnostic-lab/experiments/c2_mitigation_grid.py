#!/usr/bin/env python3
"""C2 — standard mitigations do not fix long-horizon collapse."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
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
]


def selected_cells(cfg):
    names = cfg.get("c2", {}).get("cells")
    if not names:
        return CELLS
    wanted = set(names)
    return [c for c in CELLS if c["name"] in wanted]


def main():
    args = parse_args("C2 mitigation grid")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c2_mitigation_grid")
    headline_sp = cfg["eval"].get("headline_split", 10)
    cells = selected_cells(cfg)
    results = []

    for cell in cells:
        long_vals = []
        for seed in seeds:
            seed_everything(seed)
            cell_cfg = overlay_temporal(cfg, **{k: v for k, v in cell.items() if k != "name"})
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
            long_vals.append(long[str(headline_sp)]["long_mse_mean"])
        mu, sd = mean_std(long_vals)
        rec = {
            "name": cell["name"],
            **{k: v for k, v in cell.items() if k != "name"},
            "long_mse_mean": mu,
            "long_mse_std": sd,
            "per_seed": long_vals,
        }
        results.append(rec)
        print(f"C2 | {cell['name']}: {mu:.4f} ± {sd:.4f}")

    baseline = next((r for r in results if r["name"] == "baseline"), None)
    summary = "C2 | " + "  ".join(f"{r['name']}={r['long_mse_mean']:.4f}" for r in results)
    if baseline is not None:
        summary += f"  (baseline {baseline['long_mse_mean']:.4f})"
    metrics = {"claim": "C2", "headline_split": headline_sp, "cells": results, "summary": summary}
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
