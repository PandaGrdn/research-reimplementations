#!/usr/bin/env python3
"""C3 — temporal net converges to copy-last / near-identity (motion_gap, delta_ratio).

C3 needs no training of its own once C1 has run under the same config:

    python experiments/c3_copy_detection.py --from-c1 results/c1_rollout_collapse/latest
"""

import json
import sys
from pathlib import Path

import yaml

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


TRAINING_SECTIONS = ("data", "spatial", "inference", "temporal")


def aggregate(per_seed, cfg=None):
    """Build the C3 metrics dict from per-seed records ({seed, history, split_hash})."""
    gap_curves, ratio_curves, final_gaps, final_ratios = [], [], [], []
    for rec in per_seed:
        gaps = [h.get("motion_gap") for h in rec["history"]]
        ratios = [h.get("delta_ratio") for h in rec["history"]]
        gap_curves.append(gaps)
        ratio_curves.append(ratios)
        final_gaps.append(gaps[-1] if gaps else float("nan"))
        final_ratios.append(ratios[-1] if ratios else float("nan"))

    gap_mu, gap_sd = stack_mean_std(gap_curves)
    ratio_mu, ratio_sd = stack_mean_std(ratio_curves)
    g_mu, g_sd = mean_std(final_gaps)
    r_mu, r_sd = mean_std(final_ratios)
    summary = (
        f"C3 | final motion_gap {g_mu:+.4f} ± {g_sd:.4f}  "
        f"delta_ratio {r_mu:.4f} ± {r_sd:.4f}  ({'COPY' if g_mu > 0 else 'MOTION'})"
    )
    return {
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


def records_from_c1(c1_dir, cfg, seeds):
    """C3 trains exactly the model C1 trains (same config, same seed, same
    train_temporal_pc call) and only reads the per-epoch motion_gap /
    delta_ratio that C1 already stores. So a finished C1 run contains C3.
    Refuses if the C1 run was trained under a different config or lacks a seed."""
    c1_dir = Path(c1_dir)
    c1_metrics = json.loads((c1_dir / "metrics.json").read_text())
    c1_cfg = yaml.safe_load((c1_dir / "config.yaml").read_text())
    diff = [k for k in TRAINING_SECTIONS if c1_cfg.get(k) != cfg.get(k)]
    if diff:
        raise SystemExit(
            f"--from-c1 {c1_dir}: that run's config differs from this one in {diff}; "
            f"pass the same --smoke/--overlay/--n-train/--epochs flags C1 was run with, or train C3 directly."
        )
    by_seed = {rec["seed"]: rec for rec in c1_metrics["seeds"]}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise SystemExit(f"--from-c1 {c1_dir}: C1 run has seeds {sorted(by_seed)}, missing {missing}")
    return [
        {"seed": s, "history": by_seed[s]["history"], "split_hash": by_seed[s]["split_hash"]}
        for s in seeds
    ]


def main():
    from_c1 = None
    if "--from-c1" in sys.argv:
        i = sys.argv.index("--from-c1")
        from_c1 = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = parse_args("C3 copy-last detection")
    root, cfg, device, seeds = setup(args)
    if from_c1:
        per_seed = records_from_c1(from_c1, cfg, seeds)
        run_dir = new_run_dir(root, "c3_copy_detection", arm=arm_name(cfg))
        metrics = aggregate(per_seed, cfg)
        metrics["derived_from_c1"] = str(Path(from_c1).resolve())
        finish_run(run_dir, cfg, metrics, root=root, summary=metrics["summary"])
        return
    run_dir = new_run_dir(root, "c3_copy_detection", arm=arm_name(cfg))
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
        per_seed.append({"seed": seed, "history": history, "split_hash": info["split_hash"]})

    metrics = aggregate(per_seed, cfg)
    finish_run(run_dir, cfg, metrics, root=root, summary=metrics["summary"])


if __name__ == "__main__":
    main()
