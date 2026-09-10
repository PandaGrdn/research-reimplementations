#!/usr/bin/env python3
"""C0 -- the isolation test.

The outline's step 3 ("settle the true frame vs settle the synthetic frame at
the handoff, compare convergence") had no script. This is it.

For each held-out sequence, train the model once (the usual iterative-settle
pipeline), then at the headline split point s: reconstruct r_{s-1} by warm
replaying frames 0..s-1, and settle -- to a convergence tolerance, not a fixed
iteration count -- on both the true frame I_s and the synthetic frame Î_s the
model actually substitutes in during long rollout (`validate_hierarchical_long`
with `split_fix=True`). Repeat one step later (I_{s+1} vs Î_{s+1}) to see
whether the problem is the first synthetic frame or accumulates.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import arm_name, build_temporal, ensure_dictionary, load_data, parse_args, setup, train_temporal_pc
from src.inference import settle_grounded, settle_info
from src.metrics import pair_stats, saturation_frac
from src.rollout import validate_hierarchical_long
from src.spatial_pc import recon_mse
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def replay_to_split(seq, r_init, deconvs, cfg, device, n_frames):
    """Warm-start settle over frames 0..n_frames-1 to approximate r_{n_frames-1}.

    NOTE: this is an approximation of the code the real validator would be
    holding at that point. `validate_hierarchical_long` settles each
    teacher-forced frame with `settle_with_temporal_prior`, which also pulls r
    toward the temporal net's r_pred (and, if lambda_slow>0, toward r_prev1).
    This replay uses plain `settle_grounded` (spatial-only, no temporal pull),
    chained warm-start frame to frame starting from the fixed `r_init` -- the
    code the dictionary alone would settle to given the observed frames,
    without the temporal prior's regularizing pull. Close but not identical.
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    r_curr = None
    for t in range(n_frames):
        I = seq[t] - seq[t].mean()
        if I.ndim == 3:
            I = I.unsqueeze(0)
        warm = r_curr if r_curr is not None else r_init
        r_curr, _ = settle_grounded(
            I, r_init, deconvs, inf["alpha"], inf["lr_r"], inf["sigma_2"], inf["num_epochs_inner"], num_layers,
            r_warm=warm, init_noise=inf.get("init_noise", 0.01), use_prior=inf.get("use_prior", True),
        )
    return r_curr


def _perturb(r, std):
    return [ri + torch.randn_like(ri) * std for ri in r]


def frame_diag(I, r_warm, deconvs, cfg, conv_tol, max_iters, perturb_std):
    """Settle on `I` from `r_warm` to `conv_tol` (or `max_iters`); also settle
    a second time from a perturbed warm start to measure two-init consistency.
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]

    r_a, n_used = settle_grounded(
        I, r_warm, deconvs, inf["alpha"], inf["lr_r"], inf["sigma_2"], max_iters, num_layers,
        r_warm=r_warm, conv_tol=conv_tol, init_noise=inf.get("init_noise", 0.01), use_prior=inf.get("use_prior", True),
    )
    info = settle_info(I, r_a, deconvs, inf["alpha"], inf["sigma_2"], num_layers, use_prior=inf.get("use_prior", True))
    mse, _, _ = recon_mse(I, r_a, deconvs)

    r_b, _ = settle_grounded(
        I, r_warm, deconvs, inf["alpha"], inf["lr_r"], inf["sigma_2"], max_iters, num_layers,
        r_warm=_perturb(r_warm, perturb_std), conv_tol=conv_tol, init_noise=inf.get("init_noise", 0.01),
        use_prior=inf.get("use_prior", True),
    )
    two_init_cos = pair_stats(r_a, r_b)["cos"]

    return {
        "r": r_a,
        "iters": n_used,
        "total_energy": info["total_energy"],
        "energy_per_layer": info["energy"],
        "mean_abs_dr": info["mean_abs_dr"],
        "recon_mse": mse,
        "two_init_cos": two_init_cos,
    }


def pair_diag(I_true, I_syn, r_warm, deconvs, cfg, conv_tol, max_iters, perturb_std):
    d_true = frame_diag(I_true, r_warm, deconvs, cfg, conv_tol, max_iters, perturb_std)
    d_syn = frame_diag(I_syn, r_warm, deconvs, cfg, conv_tol, max_iters, perturb_std)
    cross = pair_stats(d_true["r"], d_syn["r"])
    return {
        "true": d_true,
        "synthetic": d_syn,
        "cross_cos": cross["cos"],
        "cross_rel": cross["rel"],
        "pixel_mse_true_vs_syn": float(torch.mean((I_true - I_syn) ** 2).item()),
        "saturation": saturation_frac(I_syn),
    }


def strip_pair(d):
    out = {k: v for k, v in d.items() if k not in ("true", "synthetic")}
    out["true"] = {k: v for k, v in d["true"].items() if k != "r"}
    out["synthetic"] = {k: v for k, v in d["synthetic"].items() if k != "r"}
    return out


def isolation_test_for_seq(seq, r_init, deconvs, model, cfg, device, headline_sp, conv_tol, max_iters, perturb_std):
    seq_dev = seq.to(device)
    last = validate_hierarchical_long(
        seq_dev,
        r_init,
        cfg["inference"]["num_epochs_inner"],
        cfg["spatial"]["num_layers"],
        cfg["inference"]["sigma_2"],
        cfg["inference"]["alpha"],
        cfg["inference"]["lr_r"],
        deconvs,
        model,
        split_point=headline_sp,
        split_fix=True,
        use_prior=cfg["inference"].get("use_prior", True),
        temporal_prior_weight=cfg["inference"].get("temporal_prior_weight", 0.01),
        log=lambda *a, **k: None,
    )
    # true_frames / pred_frames are appended for every t>=1, so index k <-> t=k+1.
    I_s_true = last["true_frames"][headline_sp - 1]
    I_s_syn = last["pred_frames"][headline_sp - 1]
    I_s1_true = last["true_frames"][headline_sp]
    I_s1_syn = last["pred_frames"][headline_sp]

    r_before = replay_to_split(seq_dev, r_init, deconvs, cfg, device, headline_sp)
    d1 = pair_diag(I_s_true, I_s_syn, r_before, deconvs, cfg, conv_tol, max_iters, perturb_std)

    r_s = d1["synthetic"]["r"]  # the code the real rollout would hand off with
    d2 = pair_diag(I_s1_true, I_s1_syn, r_s, deconvs, cfg, conv_tol, max_iters, perturb_std)
    return d1, d2


def _agg(xs):
    mu, sd = mean_std(xs)
    return {"mean": mu, "std": sd}


def _agg_side(pooled):
    return {k: _agg(v) for k, v in pooled.items()}


def main():
    args = parse_args("C0 isolation test")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c0_isolation_test", arm=arm_name(cfg))
    c0cfg = cfg.get("c0", {})
    conv_tol = c0cfg.get("conv_tol", 1e-4)
    max_iters = c0cfg.get("max_iters", 2000)
    perturb_std = c0cfg.get("perturb", 0.01)
    headline_sp = cfg["eval"].get("headline_split", 10)
    n_eval = cfg["eval"].get("n_rollout_sequences", 8)

    side_keys = ("iters", "total_energy", "mean_abs_dr", "recon_mse", "two_init_cos")

    def fresh_step():
        return {
            "true": {k: [] for k in side_keys},
            "synthetic": {k: [] for k in side_keys},
            "cross_cos": [], "cross_rel": [], "pixel_mse": [], "saturation": [],
        }

    pooled = {"step1": fresh_step(), "step2": fresh_step()}
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
        model = build_temporal(cfg, r_init, device)
        model, _, _ = train_temporal_pc(train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print)

        n_use = min(n_eval, len(test))
        seed_seqs = []
        for seq in test[:n_use]:
            if seq.shape[0] <= headline_sp + 1:
                continue
            d1, d2 = isolation_test_for_seq(seq, r_init, deconvs, model, cfg, device, headline_sp, conv_tol, max_iters, perturb_std)
            seed_seqs.append({"step1": strip_pair(d1), "step2": strip_pair(d2)})
            for step_key, d in (("step1", d1), ("step2", d2)):
                for side in ("true", "synthetic"):
                    for k in side_keys:
                        pooled[step_key][side][k].append(d[side][k])
                pooled[step_key]["cross_cos"].append(d["cross_cos"])
                pooled[step_key]["cross_rel"].append(d["cross_rel"])
                pooled[step_key]["pixel_mse"].append(d["pixel_mse_true_vs_syn"])
                pooled[step_key]["saturation"].append(d["saturation"])
        per_seed.append({"seed": seed, "split_hash": info["split_hash"], "sequences": seed_seqs})

    def build_step(step_key):
        p = pooled[step_key]
        return {
            "true": _agg_side(p["true"]),
            "synthetic": _agg_side(p["synthetic"]),
            "cross_cos": _agg(p["cross_cos"]),
            "cross_rel": _agg(p["cross_rel"]),
            "pixel_mse_true_vs_syn": _agg(p["pixel_mse"]),
            "saturation": _agg(p["saturation"]),
            "n": len(p["cross_cos"]),
        }

    step1 = build_step("step1")
    step2 = build_step("step2")

    summary = (
        f"C0 | settle on true frame: E={step1['true']['total_energy']['mean']:.4f}, "
        f"iters={step1['true']['iters']['mean']:.1f}, two-init cos={step1['true']['two_init_cos']['mean']:.3f} | "
        f"on synthetic: E={step1['synthetic']['total_energy']['mean']:.4f}, "
        f"iters={step1['synthetic']['iters']['mean']:.1f}, two-init cos={step1['synthetic']['two_init_cos']['mean']:.3f} | "
        f"cos(true,syn)={step1['cross_cos']['mean']:.3f} | "
        f"pixel mse={step1['pixel_mse_true_vs_syn']['mean']:.4f} sat={step1['saturation']['mean']:.4f}"
    )
    metrics = {
        "claim": "C0",
        "seeds": per_seed,
        "step1": step1,
        "step2": step2,
        "headline_split": headline_sp,
        "conv_tol": conv_tol,
        "max_iters": max_iters,
        "perturb": perturb_std,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
