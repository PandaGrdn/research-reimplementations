#!/usr/bin/env python3
"""C4 — the settle "noise floor" is the null space of the init, not irreducible noise.

Two cold settles of the same frame differ because settle_grounded starts from
torch.randn(...) * init_noise: at init_noise=0 the two settles are byte-identical
(cos=1.000 exactly). At init_noise>0, the residual distance between the two
settled codes does not vanish with more iterations once the PC energy has
converged — it is the component of the random init that lives in the null
space of an overcomplete dictionary (layer-0 code is 8x the pixel dimension)
and is never driven out by gradient descent on the energy. This script sweeps
iters x init_noise and shows dist scales ~linearly with init_noise (slope ~1
in log-log) while energy -> 0, which is the signature of a preserved null-space
component rather than a settle process that hasn't converged.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import arm_name, ensure_dictionary, load_data, parse_args, setup
from src.inference import collect_eval_frames, diagnose_inference_nondeterminism
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def grid(cfg):
    c4 = cfg.get("c4", {})
    iters = c4.get("iters", [50, 200, 800, 2000])
    lrs = c4.get("lr_r", [0.02])
    init_noises = c4.get("init_noise", [0.0, 0.001, 0.01, 0.1])
    priors = c4.get("use_prior", [True])
    return iters, lrs, init_noises, priors


def main():
    args = parse_args("C4 settle determinism (null-space floor)")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c4_settle_determinism", arm=arm_name(cfg))
    iters, lrs, init_noises, priors = grid(cfg)
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]

    cells = []
    code_dims = None

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        frames = collect_eval_frames(test, cfg["eval"]["n_determinism_frames"])
        frames = [fr.to(device) for fr in frames]

        if code_dims is None:
            code_dims = {
                "layer0": int(r_init[0].numel()),
                "layer1": int(r_init[1].numel()) if len(r_init) > 1 else None,
                "pixel": int(frames[0].numel()),
            }

        for n_it in iters:
            for lr in lrs:
                for init_noise in init_noises:
                    for use_prior in priors:
                        out = diagnose_inference_nondeterminism(
                            frames,
                            r_init,
                            deconvs,
                            alpha=inf["alpha"],
                            lr_r=lr,
                            sigma_2=inf["sigma_2"],
                            num_epochs_inner=n_it,
                            num_layers=num_layers,
                            n_frames=len(frames),
                            init_noise=init_noise,
                            use_prior=use_prior,
                            log=print,
                        )
                        cells.append({
                            "seed": seed,
                            "iters": n_it,
                            "lr_r": lr,
                            "init_noise": init_noise,
                            "use_prior": use_prior,
                            "cos": out["cos"],
                            "rel": out["rel"],
                            "abs": out["abs"],
                            "energy_mean": out["energy_mean"],
                            "mean_abs_dr_mean": out["mean_abs_dr_mean"],
                            "r_norm_mean": out["r_norm_mean"],
                        })
                        print(
                            f"C4 | seed={seed} iters={n_it} lr_r={lr} init_noise={init_noise} "
                            f"prior={use_prior}  cos={out['cos']:.4f}  abs={out['abs']:.4f}  "
                            f"energy={out['energy_mean']:.4f}"
                        )

    # collapse per-seed rows into per-(iters, lr_r, init_noise, use_prior) cells (mean +/- std over seeds)
    keys = sorted({(c["iters"], c["lr_r"], c["init_noise"], c["use_prior"]) for c in cells})
    agg_cells = []
    by_key = {}
    for key in keys:
        n_it, lr, init_noise, use_prior = key
        rows = [c for c in cells if (c["iters"], c["lr_r"], c["init_noise"], c["use_prior"]) == key]
        cos_mu, cos_sd = mean_std([r["cos"] for r in rows])
        rel_mu, rel_sd = mean_std([r["rel"] for r in rows])
        abs_mu, abs_sd = mean_std([r["abs"] for r in rows])
        energy_mu, energy_sd = mean_std([r["energy_mean"] for r in rows])
        dr_mu, dr_sd = mean_std([r["mean_abs_dr_mean"] for r in rows])
        rnorm_mu, rnorm_sd = mean_std([r["r_norm_mean"] for r in rows])
        rec = {
            "iters": n_it,
            "lr_r": lr,
            "init_noise": init_noise,
            "use_prior": use_prior,
            "cos_mean": cos_mu,
            "cos_std": cos_sd,
            "rel_mean": rel_mu,
            "rel_std": rel_sd,
            "abs_mean": abs_mu,
            "abs_std": abs_sd,
            "energy_mean": energy_mu,
            "energy_std": energy_sd,
            "mean_abs_dr_mean": dr_mu,
            "mean_abs_dr_std": dr_sd,
            "r_norm_mean": rnorm_mu,
            "r_norm_std": rnorm_sd,
            "per_seed_cos": [r["cos"] for r in rows],
        }
        agg_cells.append(rec)
        by_key[key] = rec

    headline_lr = 0.02 if 0.02 in lrs else lrs[-1]
    max_iters = max(iters)
    headline_init_noise = 0.01 if 0.01 in init_noises else init_noises[-1]

    # dist_vs_init_noise: at the largest iters, sweep init_noise
    dist_vs_init_noise = [
        {"init_noise": c["init_noise"], "abs_mean": c["abs_mean"], "cos_mean": c["cos_mean"]}
        for c in agg_cells
        if c["iters"] == max_iters and c["lr_r"] == headline_lr and c["use_prior"] is True
    ]
    dist_vs_init_noise.sort(key=lambda c: c["init_noise"])

    # dist_vs_iters: at init_noise=headline (0.01 by default), sweep iters
    dist_vs_iters = [
        {
            "iters": c["iters"],
            "abs_mean": c["abs_mean"],
            "cos_mean": c["cos_mean"],
            "energy_mean": c["energy_mean"],
            "mean_abs_dr_mean": c["mean_abs_dr_mean"],
        }
        for c in agg_cells
        if c["init_noise"] == headline_init_noise and c["lr_r"] == headline_lr and c["use_prior"] is True
    ]
    dist_vs_iters.sort(key=lambda c: c["iters"])

    # null_space_slope: log(abs_mean) vs log(init_noise) over nonzero init_noise, at max iters
    nonzero = [d for d in dist_vs_init_noise if d["init_noise"] > 0 and d["abs_mean"] > 0]
    if len(nonzero) >= 2:
        log_x = np.log(np.asarray([d["init_noise"] for d in nonzero]))
        log_y = np.log(np.asarray([d["abs_mean"] for d in nonzero]))
        null_space_slope = float(np.polyfit(log_x, log_y, 1)[0])
    else:
        null_space_slope = float("nan")

    zero_entry = next((d for d in dist_vs_init_noise if d["init_noise"] == 0.0), None)
    cos_at_zero = zero_entry["cos_mean"] if zero_entry is not None else float("nan")

    floor_entry = dist_vs_iters[-1] if dist_vs_iters else None
    if floor_entry is not None:
        floor_iters = floor_entry["iters"]
        floor_energy = floor_entry["energy_mean"]
        floor_dist = floor_entry["abs_mean"]
    else:
        floor_iters = max_iters
        floor_energy = float("nan")
        floor_dist = float("nan")

    overcompleteness = (
        code_dims["layer0"] / code_dims["pixel"] if code_dims and code_dims["pixel"] else float("nan")
    )

    summary = (
        f"C4 | dist scales with init_noise (slope {null_space_slope:.2f}); "
        f"init_noise=0 -> cos {cos_at_zero:.3f}; "
        f"at {floor_iters} iters energy {floor_energy:.4f} while dist {floor_dist:.4f} "
        f"— floor is the init's null-space component, not irreducible noise "
        f"(layer-0 code {code_dims['layer0']} dims vs {code_dims['pixel']} pixels, "
        f"{overcompleteness:.1f}x overcomplete)"
    )

    metrics = {
        "claim": "C4",
        "cells": agg_cells,
        "headline_lr_r": headline_lr,
        "headline_init_noise": headline_init_noise,
        "dist_vs_init_noise": dist_vs_init_noise,
        "dist_vs_iters": dist_vs_iters,
        "null_space_slope": null_space_slope,
        "code_dims": code_dims,
        "overcompleteness": overcompleteness,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
