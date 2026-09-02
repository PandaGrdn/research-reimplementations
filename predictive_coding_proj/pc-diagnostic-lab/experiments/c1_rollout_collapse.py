#!/usr/bin/env python3
"""C1 — long-horizon rollout collapses at the GT → self-prediction handoff."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
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


def main():
    args = parse_args("C1 rollout collapse")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c1_rollout_collapse")
    per_seed = []
    by_split = {str(sp): {"curves": [], "long": []} for sp in cfg["eval"]["split_points"]}

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

    splits = {}
    for sp, payload in by_split.items():
        mu, sd = stack_mean_std(payload["curves"])
        long_mu, long_sd = mean_std(payload["long"])
        splits[sp] = {
            "mse_per_frame_mean": mu,
            "mse_per_frame_std": sd,
            "long_mse_mean": long_mu,
            "long_mse_std": long_sd,
        }

    headline_sp = str(cfg["eval"].get("headline_split", 10))
    h = splits.get(headline_sp) or next(iter(splits.values()))
    summary = (
        f"C1 | split={headline_sp} long MSE {h['long_mse_mean']:.4f} ± {h['long_mse_std']:.4f} "
        f"| n_seeds={len(seeds)}"
    )
    metrics = {
        "claim": "C1",
        "seeds": per_seed,
        "split_points": splits,
        "headline_split": headline_sp,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
