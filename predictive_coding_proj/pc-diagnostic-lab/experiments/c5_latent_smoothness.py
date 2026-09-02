#!/usr/bin/env python3
"""C5 — sparse latent trajectory is temporally non-smooth."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import ensure_dictionary, load_data, parse_args, setup
from src.inference import collect_unrelated_frames, diagnose_latent_smoothness
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def run_condition(cfg, test, r_init, deconvs, device, warm, label, seed):
    inf = cfg["inference"]
    n_seq = cfg["eval"]["n_pair_sequences"]
    seqs = test[:n_seq]
    unrelated = collect_unrelated_frames(test[n_seq:], cfg["eval"]["n_unrelated_frames"])
    if not unrelated:
        unrelated = collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])
    all_cons_cos, all_un_cos = [], []
    frac_s, cons_cos_s, un_cos_s, cons_abs_s = [], [], [], []
    for seq in seqs:
        out = diagnose_latent_smoothness(
            seq.to(device),
            [fr.to(device) for fr in unrelated],
            r_init,
            deconvs,
            alpha=inf["alpha"],
            lr_r=inf["lr_r"],
            sigma_2=inf["sigma_2"],
            num_epochs_inner=inf["num_epochs_inner"],
            num_layers=cfg["spatial"]["num_layers"],
            max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
            warm_start=warm,
            label=label,
            init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True),
            log=print,
        )
        frac_s.append(out["frac_rel"])
        cons_cos_s.append(out["cons_cos"])
        un_cos_s.append(out["un_cos"])
        cons_abs_s.append(out["cons_abs"])
        all_cons_cos.extend(out["cons_cos_list"])
        all_un_cos.extend(out["un_cos_list"])
    return {
        "tag": label,
        "warm_start": warm,
        "frac_rel_mean": mean_std(frac_s)[0],
        "frac_rel_std": mean_std(frac_s)[1],
        "cons_cos_mean": mean_std(cons_cos_s)[0],
        "cons_cos_std": mean_std(cons_cos_s)[1],
        "un_cos_mean": mean_std(un_cos_s)[0],
        "un_cos_std": mean_std(un_cos_s)[1],
        "cons_abs_mean": mean_std(cons_abs_s)[0],
        "cons_abs_std": mean_std(cons_abs_s)[1],
        "n_cons": len(all_cons_cos),
        "n_un": len(all_un_cos),
        "cons_cos_list": all_cons_cos,
        "un_cos_list": all_un_cos,
    }


def main():
    args = parse_args("C5 latent smoothness")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c5_latent_smoothness")
    cold_frac, warm_frac = [], []
    per_seed = []
    pooled_cold_cons, pooled_cold_un = [], []
    pooled_warm_cons, pooled_warm_un = [], []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        cold = run_condition(cfg, test, r_init, deconvs, device, False, "cold-start", seed)
        warm = run_condition(cfg, test, r_init, deconvs, device, True, "warm-start", seed)
        cold_frac.append(cold["frac_rel_mean"])
        warm_frac.append(warm["frac_rel_mean"])
        pooled_cold_cons.extend(cold["cons_cos_list"])
        pooled_cold_un.extend(cold["un_cos_list"])
        pooled_warm_cons.extend(warm["cons_cos_list"])
        pooled_warm_un.extend(warm["un_cos_list"])
        per_seed.append({"seed": seed, "cold": cold, "warm": warm, "split_hash": info["split_hash"]})

    c_mu, c_sd = mean_std(cold_frac)
    w_mu, w_sd = mean_std(warm_frac)
    n_cons = len(pooled_cold_cons)
    if n_cons < 200 and not args.smoke:
        print(f"WARNING: only {n_cons} consecutive pairs (need ≥200 to report C5).")
    summary = (
        f"C5 | consec/unrelated rel  cold={c_mu:.3f} ± {c_sd:.3f}  "
        f"warm={w_mu:.3f} ± {w_sd:.3f}  n_cons={per_seed[0]['cold']['n_cons'] if per_seed else 0}"
    )
    metrics = {
        "claim": "C5",
        "seeds": per_seed,
        "cold": {
            "frac_rel_mean": c_mu,
            "frac_rel_std": c_sd,
            "cons_cos_list": pooled_cold_cons,
            "un_cos_list": pooled_cold_un,
        },
        "warm": {
            "frac_rel_mean": w_mu,
            "frac_rel_std": w_sd,
            "cons_cos_list": pooled_warm_cons,
            "un_cos_list": pooled_warm_un,
        },
        "headline_ratio": w_mu,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
