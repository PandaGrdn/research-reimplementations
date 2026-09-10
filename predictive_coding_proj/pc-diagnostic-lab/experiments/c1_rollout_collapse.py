#!/usr/bin/env python3
"""C1 — long-horizon rollout collapses at the GT → self-prediction handoff."""

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
    parse_args,
    setup,
    train_temporal_pc,
)
from src.metrics import stack_mean_std
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def collapse_onset(mse_curve, copy_last_curve, split_point):
    """Diagnostic: is the collapse instant at the handoff, or gradual?

    mse_curve[k] / copy_last_curve[k] is the prediction for raw sequence frame
    t=k+1. `onset_frame` is the first t >= split_point where the model's MSE
    exceeds the copy-last floor at that frame (None if it never does over the
    curve). `ratio_after_split` / `ratio_last3` compare model/copy-last
    averaged over the 3 frames right after the handoff vs. the last 3 frames
    of the rollout — close values mean the collapse is immediate and does not
    get worse, growing values mean it compounds.
    """
    start = max(split_point - 1, 0)
    onset = None
    for k in range(start, len(mse_curve)):
        if k < len(copy_last_curve) and mse_curve[k] > copy_last_curve[k]:
            onset = k + 1 
            break

    def _ratio(mse_slice, floor_slice):
        if not mse_slice or not floor_slice:
            return float("nan")
        m = sum(mse_slice) / len(mse_slice)
        f = sum(floor_slice) / len(floor_slice)
        return m / (f + 1e-8)

    after_mse = mse_curve[start:start + 3]
    after_floor = copy_last_curve[start:start + 3]
    last_mse = mse_curve[-3:]
    last_floor = copy_last_curve[-3:]
    return {
        "onset_frame": onset,
        "ratio_after_split": _ratio(after_mse, after_floor),
        "ratio_last3": _ratio(last_mse, last_floor),
    }


def main():
    args = parse_args("C1 rollout collapse")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c1_rollout_collapse", arm=arm_name(cfg))
    per_seed = []
    by_split = {
        str(sp): {
            "curves": [],
            "long": [],
            "copy_last_long": [],
            "mean_frame_long": [],
            "copy_last_curve": [],
            "mean_frame_curve": [],
        }
        for sp in cfg["eval"]["split_points"]
    }
    tf_mse_vals, tf_copy_last_vals = [], []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        model = build_temporal(cfg, r_init, device)
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model, history, _ = train_temporal_pc(
            train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print, run_dir=seed_dir
        )
        long = eval_long_rollouts(test, r_init, deconvs, model, cfg, device, log=print, run_dir=seed_dir)
        rec = {"seed": seed, "history": history, "long": long, "split_hash": info["split_hash"]}
        per_seed.append(rec)
        for sp, payload in long.items():
            by_split[sp]["curves"].append(payload["mse_per_frame_mean"])
            by_split[sp]["long"].append(payload["long_mse_mean"])
            by_split[sp]["copy_last_long"].append(payload["copy_last_long_mse_mean"])
            by_split[sp]["mean_frame_long"].append(payload["mean_frame_long_mse_mean"])
            by_split[sp]["copy_last_curve"].append(payload["copy_last_mse_per_frame_mean"])
            by_split[sp]["mean_frame_curve"].append(payload["mean_frame_mse_per_frame_mean"])
        if history:
            last_epoch = history[-1]
            if "val_mse" in last_epoch:
                tf_mse_vals.append(last_epoch["val_mse"])
            if "copy_last_mse" in last_epoch:
                tf_copy_last_vals.append(last_epoch["copy_last_mse"])

    splits = {}
    for sp, payload in by_split.items():
        mu, sd = stack_mean_std(payload["curves"])
        long_mu, long_sd = mean_std(payload["long"])
        cl_long_mu, cl_long_sd = mean_std(payload["copy_last_long"])
        mf_long_mu, mf_long_sd = mean_std(payload["mean_frame_long"])
        cl_curve_mu, _ = stack_mean_std(payload["copy_last_curve"])
        mf_curve_mu, _ = stack_mean_std(payload["mean_frame_curve"])
        splits[sp] = {
            "mse_per_frame_mean": mu,
            "mse_per_frame_std": sd,
            "long_mse_mean": long_mu,
            "long_mse_std": long_sd,
            "copy_last_long_mse_mean": cl_long_mu,
            "copy_last_long_mse_std": cl_long_sd,
            "mean_frame_long_mse_mean": mf_long_mu,
            "mean_frame_long_mse_std": mf_long_sd,
            "copy_last_mse_per_frame_mean": cl_curve_mu,
            "mean_frame_mse_per_frame_mean": mf_curve_mu,
        }

    headline_sp = str(cfg["eval"].get("headline_split", 10))
    h = splits.get(headline_sp) or next(iter(splits.values()))
    if headline_sp not in splits:
        headline_sp = next(iter(splits.keys()))

    onset = collapse_onset(h["mse_per_frame_mean"], h["copy_last_mse_per_frame_mean"], int(headline_sp))

    tf_mse_mean, tf_mse_std = mean_std(tf_mse_vals)
    tf_copy_last_mean, tf_copy_last_std = mean_std(tf_copy_last_vals)

    summary = (
        f"C1 | split={headline_sp} long MSE {h['long_mse_mean']:.4f} ± {h['long_mse_std']:.4f} "
        f"| copy-last floor {h['copy_last_long_mse_mean']:.4f} | mean-frame floor {h['mean_frame_long_mse_mean']:.4f} "
        f"| tf MSE at last epoch {tf_mse_mean:.4f} (copy-last {tf_copy_last_mean:.4f}) | n_seeds={len(seeds)}"
    )
    metrics = {
        "claim": "C1",
        "seeds": per_seed,
        "split_points": splits,
        "headline_split": headline_sp,
        "tf_mse_mean": tf_mse_mean,
        "tf_mse_std": tf_mse_std,
        "tf_copy_last_mean": tf_copy_last_mean,
        "tf_copy_last_std": tf_copy_last_std,
        "collapse_onset": onset,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
