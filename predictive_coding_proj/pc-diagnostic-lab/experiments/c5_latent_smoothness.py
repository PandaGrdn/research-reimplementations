#!/usr/bin/env python3
"""C5 — code smoothness vs the pixel-space reference.

Consecutive-frame code cosine/distance looks bad in isolation, but Moving MNIST
pixels are ALSO not much smoother than "unrelated frame" pixels (translation
means every frame looks equally different from a random other frame in raw L2
terms). This script computes the same consecutive-vs-unrelated ratio for four
conditions — cold-start code, warm-start code, warm-start with zero init noise,
and the pixel-space frames themselves (no settle) — using the exact same
consecutive pairs and the exact same seeded subsample of unrelated pairs for
every condition, so "code vs pixel" is a paired, apples-to-apples comparison
rather than two different measurement protocols.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.experiment import arm_name, ensure_dictionary, load_data, parse_args, setup
from src.inference import collect_unrelated_frames, diagnose_latent_smoothness
from src.metrics import pair_stats
from src.utils import finish_run, mean_std, new_run_dir, seed_everything

CONDITIONS = ["cold", "warm", "warm_zero_init", "pixel"]


def unrelated_pairs(n_u, max_pairs, seed):
    pairs = [(i, j) for i in range(n_u) for j in range(i + 1, n_u)]
    if len(pairs) > max_pairs:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(pairs), generator=g)[:max_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    return pairs


def run_code_condition(seqs, unrelated, pairs, r_init, deconvs, cfg, device, warm_start, init_noise, label, num_epochs_inner):
    inf = cfg["inference"]
    acc = {"cons_cos": [], "cons_rel": [], "cons_abs": [], "un_cos": [], "un_rel": [], "un_abs": []}
    frac_rel_per_seq, cons_cos_per_seq = [], []
    for seq in seqs:
        out = diagnose_latent_smoothness(
            seq.to(device),
            [fr.to(device) for fr in unrelated],
            r_init,
            deconvs,
            alpha=inf["alpha"],
            lr_r=inf["lr_r"],
            sigma_2=inf["sigma_2"],
            num_epochs_inner=num_epochs_inner,
            num_layers=cfg["spatial"]["num_layers"],
            max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
            warm_start=warm_start,
            label=label,
            init_noise=init_noise,
            use_prior=inf.get("use_prior", True),
            unrelated_pairs=pairs,
            log=print,
        )
        acc["cons_cos"].extend(out["cons_cos_list"])
        acc["cons_rel"].extend(out["cons_rel_list"])
        acc["cons_abs"].extend(out["cons_abs_list"])
        acc["un_cos"].extend(out["un_cos_list"])
        acc["un_rel"].extend(out["un_rel_list"])
        acc["un_abs"].extend(out["un_abs_list"])
        frac_rel_per_seq.append(out["frac_rel"])
        cons_cos_per_seq.append(out["cons_cos"])
    acc["frac_rel_per_seq"] = frac_rel_per_seq
    acc["cons_cos_per_seq"] = cons_cos_per_seq
    return acc


def run_pixel_condition(seqs, unrelated, pairs):
    """Same pair protocol as run_code_condition, but on the raw (mean-centred)
    frames directly — no settle. Consecutive pairs are the (t-1, t) frames of
    each sequence; unrelated pairs are the exact same seeded `pairs` subsample."""
    acc = {"cons_cos": [], "cons_rel": [], "cons_abs": [], "un_cos": [], "un_rel": [], "un_abs": []}
    for seq in seqs:
        seq = seq.float()
        if seq.ndim == 3:
            seq = seq.unsqueeze(1)
        T = seq.shape[0]
        for t in range(1, T):
            st = pair_stats(seq[t], seq[t - 1])
            acc["cons_cos"].append(st["cos"]); acc["cons_rel"].append(st["rel"]); acc["cons_abs"].append(st["abs"])
    for i, j in pairs:
        st = pair_stats(unrelated[i], unrelated[j])
        acc["un_cos"].append(st["cos"]); acc["un_rel"].append(st["rel"]); acc["un_abs"].append(st["abs"])
    return acc


def summarize(acc):
    cons_cos = mean_std(acc["cons_cos"])[0]
    cons_rel = mean_std(acc["cons_rel"])[0]
    cons_abs = mean_std(acc["cons_abs"])[0]
    un_cos = mean_std(acc["un_cos"])[0]
    un_rel = mean_std(acc["un_rel"])[0]
    un_abs = mean_std(acc["un_abs"])[0]
    frac_rel = cons_rel / (un_rel + 1e-8)
    frac_abs = cons_abs / (un_abs + 1e-8)
    return {
        "cons_cos": cons_cos,
        "cons_rel": cons_rel,
        "cons_abs": cons_abs,
        "un_cos": un_cos,
        "un_rel": un_rel,
        "un_abs": un_abs,
        "frac_rel": frac_rel,
        "frac_abs": frac_abs,
        "n_cons": len(acc["cons_cos"]),
        "n_un": len(acc["un_cos"]),
        "cons_cos_list": acc["cons_cos"],
        "un_cos_list": acc["un_cos"],
        "cons_abs_list": acc["cons_abs"],
        "un_abs_list": acc["un_abs"],
    }


def main():
    args = parse_args("C5 latent smoothness vs pixel-space reference")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c5_latent_smoothness", arm=arm_name(cfg))
    inf = cfg["inference"]
    c5_iters = cfg.get("c5", {}).get("iters", [50, 200, 800])

    global_acc = {c: {"cons_cos": [], "cons_rel": [], "cons_abs": [], "un_cos": [], "un_rel": [], "un_abs": []} for c in CONDITIONS}
    iters_pool = {it: [] for it in c5_iters}
    iters_seed_means = {it: [] for it in c5_iters}
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        n_seq = cfg["eval"]["n_pair_sequences"]
        seqs = test[:n_seq]
        unrelated = collect_unrelated_frames(test[n_seq:], cfg["eval"]["n_unrelated_frames"])
        if not unrelated:
            unrelated = collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])
        unrelated = [fr.to(device) for fr in unrelated]
        pairs = unrelated_pairs(len(unrelated), cfg["eval"]["max_unrelated_pairs"], seed)

        seed_acc = {}
        num_epochs_inner = inf["num_epochs_inner"]
        seed_acc["cold"] = run_code_condition(
            seqs, unrelated, pairs, r_init, deconvs, cfg, device,
            warm_start=False, init_noise=inf.get("init_noise", 0.01), label="cold", num_epochs_inner=num_epochs_inner,
        )
        seed_acc["warm"] = run_code_condition(
            seqs, unrelated, pairs, r_init, deconvs, cfg, device,
            warm_start=True, init_noise=inf.get("init_noise", 0.01), label="warm", num_epochs_inner=num_epochs_inner,
        )
        seed_acc["warm_zero_init"] = run_code_condition(
            seqs, unrelated, pairs, r_init, deconvs, cfg, device,
            warm_start=True, init_noise=0.0, label="warm_zero_init", num_epochs_inner=num_epochs_inner,
        )
        seed_acc["pixel"] = run_pixel_condition(seqs, unrelated, pairs)

        for cond in CONDITIONS:
            for k in global_acc[cond]:
                global_acc[cond][k].extend(seed_acc[cond][k])

        # iteration sweep for the "warm" condition only
        for it in c5_iters:
            sweep = run_code_condition(
                seqs, unrelated, pairs, r_init, deconvs, cfg, device,
                warm_start=True, init_noise=inf.get("init_noise", 0.01), label=f"warm_iters={it}",
                num_epochs_inner=it,
            )
            iters_pool[it].extend(sweep["cons_cos"])
            iters_seed_means[it].append(mean_std(sweep["cons_cos"])[0])

        per_seed.append({
            "seed": seed,
            "split_hash": info["split_hash"],
            **{cond: summarize(seed_acc[cond]) for cond in CONDITIONS},
        })

    conditions = {cond: summarize(global_acc[cond]) for cond in CONDITIONS}

    iters_sweep = []
    for it in c5_iters:
        mu, sd = mean_std(iters_seed_means[it])
        iters_sweep.append({"iters": it, "cons_cos_mean": mu, "cons_cos_std": sd, "n": len(iters_pool[it])})
    iters_sweep.sort(key=lambda r: r["iters"])

    warm = conditions["warm"]
    pixel = conditions["pixel"]
    code_over_pixel_frac_rel = warm["frac_rel"] / (pixel["frac_rel"] + 1e-8)
    code_over_pixel_cons_cos = warm["cons_cos"] / (pixel["cons_cos"] + 1e-8)

    n_cons = warm["n_cons"]
    if n_cons < 200 and not args.smoke:
        print(f"WARNING: only {n_cons} consecutive pairs (need >=200 to report C5).")

    summary = (
        f"C5 | consec/unrelated rel: code(warm)={warm['frac_rel']:.2f} vs pixel={pixel['frac_rel']:.2f} "
        f"(ratio {code_over_pixel_frac_rel:.2f}) | cons cos code={warm['cons_cos']:.2f} "
        f"pixel={pixel['cons_cos']:.2f} | n_cons={n_cons}"
    )

    metrics = {
        "claim": "C5",
        "seeds": per_seed,
        "conditions": conditions,
        "iters_sweep": iters_sweep,
        "code_over_pixel_frac_rel": code_over_pixel_frac_rel,
        "code_over_pixel_cons_cos": code_over_pixel_cons_cos,
        "n_seeds_inference": len(seeds),
        # kept for anything expecting the old top-level shortcuts
        "cold": conditions["cold"],
        "warm": conditions["warm"],
        "headline_ratio": warm["frac_rel"],
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
