#!/usr/bin/env python3
"""C6 — prediction target is dominated by settle noise (centerpiece number)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import ensure_dictionary, load_data, parse_args, setup
from src.inference import (
    collect_eval_frames,
    collect_unrelated_frames,
    diagnose_inference_nondeterminism,
    diagnose_latent_smoothness,
)
from src.metrics import noise_floor_decomposition
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def main():
    args = parse_args("C6 noise-floor decomposition")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c6_noise_floor")
    inf = cfg["inference"]
    fracs, noise_shares, settle_abs, target_abs = [], [], [], []
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        frames = [fr.to(device) for fr in collect_eval_frames(test, cfg["eval"]["n_determinism_frames"])]
        nd = diagnose_inference_nondeterminism(
            frames,
            r_init,
            deconvs,
            alpha=inf["alpha"],
            lr_r=inf["lr_r"],
            sigma_2=inf["sigma_2"],
            num_epochs_inner=inf["num_epochs_inner"],
            num_layers=cfg["spatial"]["num_layers"],
            n_frames=len(frames),
            init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True),
            log=print,
        )
        unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
        n_seq = cfg["eval"]["n_pair_sequences"]
        cons_abs = []
        for seq in test[:n_seq]:
            sm = diagnose_latent_smoothness(
                seq.to(device),
                unrelated,
                r_init,
                deconvs,
                alpha=inf["alpha"],
                lr_r=inf["lr_r"],
                sigma_2=inf["sigma_2"],
                num_epochs_inner=inf["num_epochs_inner"],
                num_layers=cfg["spatial"]["num_layers"],
                max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
                warm_start=True,
                label="C6 warm-start",
                init_noise=inf.get("init_noise", 0.01),
                use_prior=inf.get("use_prior", True),
                log=print,
            )
            cons_abs.append(sm["cons_abs"])
        target = sum(cons_abs) / max(len(cons_abs), 1)
        decomp = noise_floor_decomposition(nd["abs"], target)
        fracs.append(decomp["predictable_fraction"])
        noise_shares.append(decomp["noise_share"])
        settle_abs.append(decomp["settle_abs"])
        target_abs.append(decomp["target_abs"])
        per_seed.append({"seed": seed, "nondet": nd, "decomp": decomp, "split_hash": info["split_hash"]})

    p_mu, p_sd = mean_std(fracs)
    n_mu, n_sd = mean_std(noise_shares)
    summary = (
        f"C6 | predictable fraction {p_mu:.3f} ± {p_sd:.3f}  "
        f"noise share {n_mu:.3f} ± {n_sd:.3f}  (2σ² vs ||r_t-r_{{t-1}}||²)"
    )
    metrics = {
        "claim": "C6",
        "seeds": per_seed,
        "predictable_fraction_mean": p_mu,
        "predictable_fraction_std": p_sd,
        "noise_share_mean": n_mu,
        "noise_share_std": n_sd,
        "settle_abs_mean": mean_std(settle_abs)[0],
        "target_abs_mean": mean_std(target_abs)[0],
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
