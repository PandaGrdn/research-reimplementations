#!/usr/bin/env python3
"""C6 — how much of the prediction target is noise, under the protocol the
pipeline actually uses.

Three ways to measure "settle noise" vs "target energy":

1. cold_independent (the old measurement): two COLD settles of the same frame
   (independent random inits) give settle_abs; target_abs comes from a
   WARM-started consecutive trajectory. This mixes two different protocols —
   labelled "protocol mismatch" below, kept only for comparison.
2. warm_independent_init (the honest version): run the WHOLE warm-started
   trajectory twice, from two different random r_0 inits. settle_abs is the
   mean over t of ||r_t^A - r_t^B||; target_abs is the mean over t of
   ||r_t^A - r_{t-1}^A||. Both numbers now come from the same warm-start
   protocol the model actually uses at inference time.
3. pipeline: identical to (2) except both trajectories start from the
   checkpointed FIXED r_init (as training/eval does) instead of a random init.
   Since settle is deterministic given the same warm-start chain, settle_abs
   should be exactly 0 — there is no "noise" in the object the real pipeline
   ever computes a target against.

noise_floor_decomposition (src/metrics.py) is not clamped, so noise_over_target
can exceed 1 (falsifying "noise is a small share of target").
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import arm_name, ensure_dictionary, load_data, parse_args, setup
from src.inference import (
    collect_eval_frames,
    collect_unrelated_frames,
    diagnose_inference_nondeterminism,
    diagnose_latent_smoothness,
    settle_trajectory,
)
from src.metrics import noise_floor_decomposition, pair_stats
from src.utils import finish_run, mean_std, new_run_dir, seed_everything

PROTOCOLS = ["cold_independent", "warm_independent_init", "pipeline"]
HEADLINE_PROTOCOL = "warm_independent_init"


def trajectory_pair_stats(seq, r_init, deconvs, inf, num_layers, fixed_start):
    """Two independent settle_trajectory rollouts of `seq`; returns (settle_abs, target_abs)
    where settle_abs = mean_t ||r_t^A - r_t^B|| and target_abs = mean_t ||r_t^A - r_{t-1}^A||."""
    kwargs = dict(
        alpha=inf["alpha"],
        lr_r=inf["lr_r"],
        sigma_2=inf["sigma_2"],
        num_epochs_inner=inf["num_epochs_inner"],
        num_layers=num_layers,
        init_noise=inf.get("init_noise", 0.01),
        use_prior=inf.get("use_prior", True),
        fixed_start=fixed_start,
    )
    r_a = settle_trajectory(seq, r_init, deconvs, **kwargs)
    r_b = settle_trajectory(seq, r_init, deconvs, **kwargs)
    T = len(r_a)
    settle_dists = [pair_stats(r_a[t], r_b[t])["abs"] for t in range(T)]
    target_dists = [pair_stats(r_a[t], r_a[t - 1])["abs"] for t in range(1, T)]
    settle_abs = sum(settle_dists) / max(len(settle_dists), 1)
    target_abs = sum(target_dists) / max(len(target_dists), 1)
    return settle_abs, target_abs


def pixel_target_abs_of(seq):
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]
    dists = [pair_stats(seq[t], seq[t - 1])["abs"] for t in range(1, T)]
    return sum(dists) / max(len(dists), 1)


def main():
    args = parse_args("C6 noise-floor decomposition under the real inference protocol")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c6_noise_floor", arm=arm_name(cfg))
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]

    per_protocol_settle = {p: [] for p in PROTOCOLS}
    per_protocol_target = {p: [] for p in PROTOCOLS}
    pixel_target_abs_list = []
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        n_seq = cfg["eval"]["n_pair_sequences"]
        seqs = [s.to(device) for s in test[:n_seq]]

        # --- protocol 1: cold_independent (old measurement, "protocol mismatch") ---
        frames = [fr.to(device) for fr in collect_eval_frames(test, cfg["eval"]["n_determinism_frames"])]
        nd = diagnose_inference_nondeterminism(
            frames, r_init, deconvs,
            alpha=inf["alpha"], lr_r=inf["lr_r"], sigma_2=inf["sigma_2"],
            num_epochs_inner=inf["num_epochs_inner"], num_layers=num_layers,
            n_frames=len(frames), init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True), log=print,
        )
        unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
        cons_abs_warm = []
        for seq in seqs:
            sm = diagnose_latent_smoothness(
                seq, unrelated, r_init, deconvs,
                alpha=inf["alpha"], lr_r=inf["lr_r"], sigma_2=inf["sigma_2"],
                num_epochs_inner=inf["num_epochs_inner"], num_layers=num_layers,
                max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"], warm_start=True,
                label="C6 warm-start", init_noise=inf.get("init_noise", 0.01),
                use_prior=inf.get("use_prior", True), log=print,
            )
            cons_abs_warm.append(sm["cons_abs"])
        per_protocol_settle["cold_independent"].append(nd["abs"])
        per_protocol_target["cold_independent"].append(sum(cons_abs_warm) / max(len(cons_abs_warm), 1))

        # --- protocols 2 & 3: two independent full-trajectory rollouts ---
        for proto, fixed_start in (("warm_independent_init", False), ("pipeline", True)):
            settle_list, target_list = [], []
            for seq in seqs:
                s_abs, t_abs = trajectory_pair_stats(seq, r_init, deconvs, inf, num_layers, fixed_start)
                settle_list.append(s_abs)
                target_list.append(t_abs)
            per_protocol_settle[proto].append(sum(settle_list) / max(len(settle_list), 1))
            per_protocol_target[proto].append(sum(target_list) / max(len(target_list), 1))

        pixel_list = [pixel_target_abs_of(seq) for seq in seqs]
        pixel_target_abs_list.append(sum(pixel_list) / max(len(pixel_list), 1))

        seed_decomps = {
            p: noise_floor_decomposition(per_protocol_settle[p][-1], per_protocol_target[p][-1])
            for p in PROTOCOLS
        }
        per_seed.append({
            "seed": seed,
            "split_hash": info["split_hash"],
            "protocols": seed_decomps,
            "pixel_target_abs": pixel_target_abs_list[-1],
        })
        for p in PROTOCOLS:
            d = seed_decomps[p]
            print(
                f"C6 | seed={seed} protocol={p}  settle_abs={d['settle_abs']:.4f}  "
                f"target_abs={d['target_abs']:.4f}  noise/target={d['noise_over_target']:.4f}  "
                f"model_consistent={d['model_consistent']}"
            )

    protocols = {}
    for p in PROTOCOLS:
        settle_mu = mean_std(per_protocol_settle[p])[0]
        target_mu = mean_std(per_protocol_target[p])[0]
        decomp = noise_floor_decomposition(settle_mu, target_mu)
        decomp["label"] = "protocol mismatch" if p == "cold_independent" else "honest"
        protocols[p] = decomp

    pixel_target_abs = mean_std(pixel_target_abs_list)[0]

    c1, c2, c3 = protocols["cold_independent"], protocols["warm_independent_init"], protocols["pipeline"]
    summary = (
        f"C6 | noise/target energy: cold_independent={c1['noise_over_target']:.4f} (mismatched), "
        f"warm_independent_init={c2['noise_over_target']:.4f}, pipeline={c3['noise_over_target']:.4f} "
        f"| model_consistent: cold_independent={c1['model_consistent']}, "
        f"warm_independent_init={c2['model_consistent']}, pipeline={c3['model_consistent']} "
        f"| pixel_target_abs={pixel_target_abs:.4f}"
    )

    metrics = {
        "claim": "C6",
        "seeds": per_seed,
        "protocols": protocols,
        "headline_protocol": HEADLINE_PROTOCOL,
        "pixel_target_abs": pixel_target_abs,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
