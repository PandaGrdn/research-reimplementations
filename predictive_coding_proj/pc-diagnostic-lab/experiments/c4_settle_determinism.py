#!/usr/bin/env python3
"""C4 — iterative settle has an irreducible noise floor (same-frame determinism)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import ensure_dictionary, load_data, parse_args, setup
from src.inference import collect_eval_frames, diagnose_inference_nondeterminism
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def grid(cfg):
    c4 = cfg.get("c4", {})
    iters = c4.get("iters", [50, 100, 200, 400])
    lrs = c4.get("lr_r", [0.005, 0.02])
    priors = c4.get("use_prior", [True])
    return iters, lrs, priors


def main():
    args = parse_args("C4 settle determinism")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c4_settle_determinism")
    iters, lrs, priors = grid(cfg)
    inf = cfg["inference"]
    cells = []
    by_key = {}

    for n_it in iters:
        for lr in lrs:
            for use_prior in priors:
                key = f"iters={n_it}|lr_r={lr}|prior={use_prior}"
                cos_s, rel_s, abs_s = [], [], []
                for seed in seeds:
                    seed_everything(seed)
                    cfg["seed"] = seed
                    train, val, test, info = load_data(cfg, root, seed)
                    deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
                    frames = collect_eval_frames(test, cfg["eval"]["n_determinism_frames"])
                    frames = [fr.to(device) for fr in frames]
                    out = diagnose_inference_nondeterminism(
                        frames,
                        r_init,
                        deconvs,
                        alpha=inf["alpha"],
                        lr_r=lr,
                        sigma_2=inf["sigma_2"],
                        num_epochs_inner=n_it,
                        num_layers=cfg["spatial"]["num_layers"],
                        n_frames=len(frames),
                        init_noise=inf.get("init_noise", 0.01),
                        use_prior=use_prior,
                        log=print,
                    )
                    cos_s.append(out["cos"])
                    rel_s.append(out["rel"])
                    abs_s.append(out["abs"])
                cos_mu, cos_sd = mean_std(cos_s)
                rel_mu, rel_sd = mean_std(rel_s)
                abs_mu, abs_sd = mean_std(abs_s)
                rec = {
                    "iters": n_it,
                    "lr_r": lr,
                    "use_prior": use_prior,
                    "cos_mean": cos_mu,
                    "cos_std": cos_sd,
                    "rel_mean": rel_mu,
                    "rel_std": rel_sd,
                    "abs_mean": abs_mu,
                    "abs_std": abs_sd,
                    "per_seed_cos": cos_s,
                }
                cells.append(rec)
                by_key[key] = rec
                print(f"C4 | {key}  cos={cos_mu:.4f} ± {cos_sd:.4f}  rel={rel_mu:.4f}")

    # headline: lr_r=0.02 curve vs iterations (matches the notebook 0.73→0.93 story)
    headline_lr = 0.02 if 0.02 in lrs else lrs[-1]
    curve = [c for c in cells if c["lr_r"] == headline_lr and c["use_prior"] is True]
    curve = sorted(curve, key=lambda c: c["iters"])
    floor = curve[-1] if curve else cells[-1]
    summary = (
        f"C4 | cos vs iters (lr_r={headline_lr}): "
        + " → ".join(f"{c['iters']}:{c['cos_mean']:.3f}" for c in curve)
        + f"  residual floor cos={floor['cos_mean']:.3f} rel={floor['rel_mean']:.3f}"
    )
    metrics = {
        "claim": "C4",
        "cells": cells,
        "headline_lr_r": headline_lr,
        "cos_vs_iters": curve,
        "residual_floor": floor,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
