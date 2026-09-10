#!/usr/bin/env python3
"""C3 — temporal net converges to copy-last / near-identity (motion_gap, delta_ratio)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    parse_args,
    setup,
    train_temporal_pc,
)
from src.metrics import stack_mean_std
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def main():
    args = parse_args("C3 copy-last detection")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c3_copy_detection", arm=arm_name(cfg))
    gap_curves, ratio_curves, final_gaps, final_ratios = [], [], [], []
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        model = build_temporal(cfg, r_init, device)
        model, history, last = train_temporal_pc(
            train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print, run_dir=run_dir / f"seed{seed}"
        )
        gaps = [h.get("motion_gap") for h in history]
        ratios = [h.get("delta_ratio") for h in history]
        gap_curves.append(gaps)
        ratio_curves.append(ratios)
        final_gaps.append(gaps[-1] if gaps else float("nan"))
        final_ratios.append(ratios[-1] if ratios else float("nan"))
        per_seed.append({"seed": seed, "history": history, "split_hash": info["split_hash"]})

    gap_mu, gap_sd = stack_mean_std(gap_curves)
    ratio_mu, ratio_sd = stack_mean_std(ratio_curves)
    g_mu, g_sd = mean_std(final_gaps)
    r_mu, r_sd = mean_std(final_ratios)
    summary = (
        f"C3 | final motion_gap {g_mu:+.4f} ± {g_sd:.4f}  "
        f"delta_ratio {r_mu:.4f} ± {r_sd:.4f}  ({'COPY' if g_mu > 0 else 'MOTION'})"
    )
    metrics = {
        "claim": "C3",
        "seeds": per_seed,
        "motion_gap_mean": gap_mu,
        "motion_gap_std": gap_sd,
        "delta_ratio_mean": ratio_mu,
        "delta_ratio_std": ratio_sd,
        "final_motion_gap_mean": g_mu,
        "final_motion_gap_std": g_sd,
        "final_delta_ratio_mean": r_mu,
        "final_delta_ratio_std": r_sd,
        "copy_last": g_mu > 0,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
