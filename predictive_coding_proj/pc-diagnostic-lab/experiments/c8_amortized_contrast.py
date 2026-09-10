#!/usr/bin/env python3
"""C8 — one variable: the inference procedure.

Every arm uses the SAME frozen pretrained dictionary, the SAME
`train_temporal_pc` loop and the SAME evaluators (`validate_hierarchical`,
`eval_long_rollouts`) via their `encoder=` hook. Only how the per-frame code
is produced changes:

  iterative              -- current settle (encoder=None)
  amortized_init_settle  -- encoder output as init, then a few plain settle steps
  amortized               -- encoder only

This isolates the inference procedure from the five things the old C8 changed
at once (amortized vs iterative, layer count, dense vs sparse, a no-op
slowness gradient, a different temporal loop).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.amortized import build_encoder, make_encoder_fn, make_init_settle_fn, train_encoder
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
from src.inference import collect_eval_frames, collect_unrelated_frames, settle_grounded, settle_info
from src.rollout import validate_hierarchical
from src.spatial_pc import recon_mse
from src.metrics import pair_stats, summarize_pair_lists
from src.predictability import encode_sequences, make_settle_encoder, pixel_codes, predictability_r2
from src.utils import finish_run, mean_std, new_run_dir, seed_everything

ARM_ORDER = ["iterative", "amortized_init_settle", "amortized"]
ARM_LABEL = {"iterative": "iter", "amortized_init_settle": "init+settle", "amortized": "amort"}


def _reset(fn):
    r = getattr(fn, "reset", None)
    if r is not None:
        r()


def make_iterative_encode_fn(cfg, r_init, deconvs):
    """Stateful callable f(I) -> list[r], warm-started frame to frame like C5's
    warm-start diagnostic. This gives the iterative arm the SAME `f(I)->r`
    interface as the amortized arms so smoothness pairs are computed by the
    exact same code path (same pair indices) across all arms.
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    state = {"r_prev": None}

    def f(I):
        r, _ = settle_grounded(
            I,
            r_init,
            deconvs,
            inf["alpha"],
            inf["lr_r"],
            inf["sigma_2"],
            inf["num_epochs_inner"],
            num_layers,
            r_warm=state["r_prev"],
            init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True),
        )
        state["r_prev"] = [ri.detach().clone() for ri in r]
        return r

    def reset():
        state["r_prev"] = None

    f.reset = reset
    return f


def pixel_reference(seqs, unrelated_frames, max_unrelated_pairs, generator=None):
    """Pixel-space consecutive vs unrelated reference, computed once (arm-independent)."""
    cons_cos, cons_rel = [], []
    for seq in seqs:
        for t in range(1, seq.shape[0]):
            s = pair_stats(seq[t], seq[t - 1])
            cons_cos.append(s["cos"])
            cons_rel.append(s["rel"])
    un_cos, un_rel = [], []
    pairs = [(i, j) for i in range(len(unrelated_frames)) for j in range(i + 1, len(unrelated_frames))]
    if len(pairs) > max_unrelated_pairs:
        g = generator if generator is not None else torch.Generator(device="cpu")
        idx = torch.randperm(len(pairs), generator=g)[:max_unrelated_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        s = pair_stats(unrelated_frames[i], unrelated_frames[j])
        un_cos.append(s["cos"])
        un_rel.append(s["rel"])
    cons = summarize_pair_lists(cons_cos, cons_rel, cons_rel)
    un = summarize_pair_lists(un_cos, un_rel, un_rel)
    frac_rel = cons["rel"] / (un["rel"] + 1e-8)
    return {"cons_cos": cons["cos"], "un_cos": un["cos"], "cons_rel": cons["rel"], "un_rel": un["rel"], "frac_rel": frac_rel}


def smoothness_via_encode_fn(encode_fn, seqs, unrelated_frames, max_unrelated_pairs, generator=None):
    """SAME pair structure (sequences, frame indices, unrelated-frame sampling)
    used for every arm -- only `encode_fn` differs.
    """
    cons_cos, cons_rel, cons_abs = [], [], []
    for seq in seqs:
        _reset(encode_fn)
        rs = []
        for t in range(seq.shape[0]):
            I = seq[t]
            if I.ndim == 3:
                I = I.unsqueeze(0)
            rs.append(encode_fn(I))
        for t in range(1, seq.shape[0]):
            s = pair_stats(rs[t], rs[t - 1])
            cons_cos.append(s["cos"])
            cons_rel.append(s["rel"])
            cons_abs.append(s["abs"])
    u_rs = []
    for I in unrelated_frames:
        _reset(encode_fn)
        x = I.unsqueeze(0) if I.ndim == 3 else I
        u_rs.append(encode_fn(x))
    un_cos, un_rel, un_abs = [], [], []
    pairs = [(i, j) for i in range(len(u_rs)) for j in range(i + 1, len(u_rs))]
    if len(pairs) > max_unrelated_pairs:
        g = generator if generator is not None else torch.Generator(device="cpu")
        idx = torch.randperm(len(pairs), generator=g)[:max_unrelated_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        s = pair_stats(u_rs[i], u_rs[j])
        un_cos.append(s["cos"])
        un_rel.append(s["rel"])
        un_abs.append(s["abs"])
    cons = summarize_pair_lists(cons_cos, cons_rel, cons_abs)
    un = summarize_pair_lists(un_cos, un_rel, un_abs)
    frac_rel = cons["rel"] / (un["rel"] + 1e-8)
    return {
        "cons_cos": cons["cos"],
        "cons_rel": cons["rel"],
        "cons_abs": cons["abs"],
        "un_cos": un["cos"],
        "un_rel": un["rel"],
        "un_abs": un["abs"],
        "frac_rel": frac_rel,
        "n_cons": len(cons_cos),
        "n_un": len(un_cos),
        "cons_cos_list": cons_cos,
        "un_cos_list": un_cos,
    }


def recon_and_energy(encode_fn, frames, deconvs, cfg):
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    mses, energies = [], []
    for I in frames:
        _reset(encode_fn)
        x = I.unsqueeze(0) if I.ndim == 3 else I
        r = encode_fn(x)
        mse, _, _ = recon_mse(x, r, deconvs)
        mses.append(mse)
        info = settle_info(x, r, deconvs, inf["alpha"], inf["sigma_2"], num_layers, use_prior=inf.get("use_prior", True))
        energies.append(info["total_energy"])
    return {
        "recon_mse": float(sum(mses) / max(len(mses), 1)),
        "energy": float(sum(energies) / max(len(energies), 1)),
    }


def same_frame_cos_iterative(cfg, r_init, deconvs, frames):
    """Two independent cold-start settles of the same frame (reproduces C4)."""
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    cos_list = []
    for I in frames:
        x = I.unsqueeze(0) if I.ndim == 3 else I
        r1, _ = settle_grounded(
            x, r_init, deconvs, inf["alpha"], inf["lr_r"], inf["sigma_2"], inf["num_epochs_inner"], num_layers,
            r_warm=None, init_noise=inf.get("init_noise", 0.01), use_prior=inf.get("use_prior", True),
        )
        r2, _ = settle_grounded(
            x, r_init, deconvs, inf["alpha"], inf["lr_r"], inf["sigma_2"], inf["num_epochs_inner"], num_layers,
            r_warm=None, init_noise=inf.get("init_noise", 0.01), use_prior=inf.get("use_prior", True),
        )
        cos_list.append(pair_stats(r1, r2)["cos"])
    return float(sum(cos_list) / max(len(cos_list), 1))


def same_frame_cos_deterministic(encode_fn, frames):
    cos_list = []
    for I in frames:
        x = I.unsqueeze(0) if I.ndim == 3 else I
        _reset(encode_fn)
        r1 = encode_fn(x)
        _reset(encode_fn)
        r2 = encode_fn(x)
        cos_list.append(pair_stats(r1, r2)["cos"])
    return float(sum(cos_list) / max(len(cos_list), 1))


def _c9_settings(cfg):
    """Defensive reads of the c9 config block (used to reuse src/predictability.py
    for a per-arm predictability check without hard-requiring every c9 key).
    """
    c9cfg = cfg.get("c9", {}) or {}
    return {
        "models": c9cfg.get("models", ["linear", "conv"]),
        "n_train_sequences": c9cfg.get("n_train_sequences", 24),
        "context": c9cfg.get("context", 2),
        "steps": c9cfg.get("steps", 300),
    }


def compute_predictability(encode_fn, train_seqs, test_seqs, device, models, context, steps, seed):
    """{model: {"code": r2_dict}} for one arm's encode_fn, reusing C9's machinery."""
    train_codes = encode_sequences(train_seqs, encode_fn)
    test_codes = encode_sequences(test_seqs, encode_fn)
    out = {}
    for model in models:
        r2 = predictability_r2(train_codes, test_codes, device, context=context, steps=steps, model=model, seed=seed)
        out[model] = {"code": r2}
    return out


def run_arm(arm, cfg, train, val, test, r_init, deconvs, device, encoder, refine_iters, pair_seqs, unrelated, held_out, headline_sp, seed, train_pred_seqs):
    inf = cfg["inference"]
    c9s = _c9_settings(cfg)

    if arm == "iterative":
        train_hook = None
        smooth_fn = make_iterative_encode_fn(cfg, r_init, deconvs)
        eval_fn_for_diag = make_iterative_encode_fn(cfg, r_init, deconvs)
        same_cos = same_frame_cos_iterative(cfg, r_init, deconvs, held_out)
        by_construction = False
    elif arm == "amortized_init_settle":
        fn = make_init_settle_fn(encoder, deconvs, cfg, refine_iters)
        train_hook = fn
        smooth_fn = fn
        eval_fn_for_diag = fn
        same_cos = same_frame_cos_deterministic(fn, held_out)
        by_construction = True
    elif arm == "amortized":
        fn = make_encoder_fn(encoder, device)
        train_hook = fn
        smooth_fn = fn
        eval_fn_for_diag = fn
        same_cos = same_frame_cos_deterministic(fn, held_out)
        by_construction = True
    else:
        raise ValueError(f"unknown c8 arm {arm!r}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    sm = smoothness_via_encode_fn(smooth_fn, pair_seqs, unrelated, cfg["eval"]["max_unrelated_pairs"], generator=gen)
    re = recon_and_energy(eval_fn_for_diag, held_out, deconvs, cfg)

    model = build_temporal(cfg, r_init, device)
    model, history, last_val = train_temporal_pc(
        train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print, encoder=train_hook
    )

    tf = validate_hierarchical(
        test[0].to(device),
        r_init,
        inf["num_epochs_inner"],
        cfg["spatial"]["num_layers"],
        inf["sigma_2"],
        inf["alpha"],
        inf["lr_r"],
        deconvs,
        model,
        use_prior=inf.get("use_prior", True),
        temporal_prior_weight=inf.get("temporal_prior_weight", 0.01),
        encoder=train_hook,
        log=print,
    )
    long = eval_long_rollouts(test, r_init, deconvs, model, cfg, device, split_points=[headline_sp], log=print, encoder=train_hook)
    lh = long[str(headline_sp)]

    # Predictability (R^2 vs copy-last), reusing src/predictability.py (C9's
    # machinery) on this arm's own encode function. The iterative arm uses the
    # SAME warm-start pipeline protocol as the real train/eval loop
    # (make_settle_encoder(warm=True)); the amortized arms are already
    # stateless f(I)->list[r] callables, matching encode_sequences directly.
    if arm == "iterative":
        pred_encode_fn = make_settle_encoder(deconvs, r_init, cfg, warm=True)
    else:
        pred_encode_fn = smooth_fn
    predictability = compute_predictability(
        pred_encode_fn, train_pred_seqs, pair_seqs, device, c9s["models"], c9s["context"], c9s["steps"], seed
    )

    return {
        "same_frame_cos": same_cos,
        "same_frame_cos_by_construction": by_construction,
        "smoothness": sm,
        "recon_mse": re["recon_mse"],
        "energy": re["energy"],
        "tf_mse": tf["mse"],
        "copy_last_mse": tf["copy_last_mse"],
        "mean_frame_mse": tf["mean_frame_mse"],
        "motion_gap": tf["motion_gap"],
        "delta_ratio": tf["delta_ratio"],
        "long_mse": lh["long_mse_mean"],
        "copy_last_long_mse": lh["copy_last_long_mse_mean"],
        "mean_frame_long_mse": lh["mean_frame_long_mse_mean"],
        "predictability": predictability,
    }


def main():
    args = parse_args("C8 amortized contrast")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c8_amortized_contrast", arm=arm_name(cfg))
    c8cfg = cfg.get("c8", {})
    arms = [a for a in ARM_ORDER if a in (c8cfg.get("arms") or ARM_ORDER)]
    refine_iters = c8cfg.get("refine_iters", 20)
    headline_sp = cfg["eval"].get("headline_split", 10)
    c9s = _c9_settings(cfg)
    primary_model = "conv" if "conv" in c9s["models"] else c9s["models"][0]

    per_seed = []
    pooled = {a: {"cons": [], "un": []} for a in arms}
    series = {a: {k: [] for k in (
        "same_frame_cos", "cons_cos", "recon_mse", "energy", "tf_mse", "copy_last_mse", "mean_frame_mse",
        "long_mse", "copy_last_long_mse", "mean_frame_long_mse", "motion_gap", "delta_ratio",
    )} for a in arms}
    pred_series = {a: {model: [] for model in c9s["models"]} for a in arms}
    pixel_pred_series = {model: [] for model in c9s["models"]}
    pixel_series = {"cons_cos": [], "un_cos": [], "frac_rel": []}
    by_construction = {}

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        n_seq = cfg["eval"]["n_pair_sequences"]
        pair_seqs = [s.to(device) for s in test[:n_seq]]
        unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
        held_out_src = test[n_seq:] or test
        held_out = [fr.to(device) for fr in collect_eval_frames(held_out_src, cfg["eval"]["n_determinism_frames"])]

        px = pixel_reference(pair_seqs, unrelated, cfg["eval"]["max_unrelated_pairs"], generator=torch.Generator(device="cpu").manual_seed(seed))
        pixel_series["cons_cos"].append(px["cons_cos"])
        pixel_series["un_cos"].append(px["un_cos"])
        pixel_series["frac_rel"].append(px["frac_rel"])

        # Pixel predictability reference: arm-independent (no code involved), so
        # computed once per seed rather than inside run_arm.
        train_pred_seqs = [s.to(device) for s in train[: c9s["n_train_sequences"]]]
        train_pixels = pixel_codes(train_pred_seqs)
        test_pixels = pixel_codes(pair_seqs)
        pixel_predictability = {}
        for model in c9s["models"]:
            r2 = predictability_r2(
                train_pixels, test_pixels, device, context=c9s["context"], steps=c9s["steps"], model=model, seed=seed
            )
            pixel_predictability[model] = r2
            pixel_pred_series[model].append(r2["r2_vs_copy_last"])

        encoder = None
        if "amortized" in arms or "amortized_init_settle" in arms:
            encoder = build_encoder(cfg, r_init).to(device)
            train_encoder(encoder, deconvs, train, cfg, device, r_init=r_init, log=print)

        seed_arms = {}
        for arm in arms:
            out = run_arm(
                arm, cfg, train, val, test, r_init, deconvs, device, encoder, refine_iters,
                pair_seqs, unrelated, held_out, headline_sp, seed, train_pred_seqs,
            )
            seed_arms[arm] = out
            for k in series[arm]:
                if k == "cons_cos":
                    series[arm][k].append(out["smoothness"]["cons_cos"])
                else:
                    series[arm][k].append(out[k])
            pooled[arm]["cons"].extend(out["smoothness"]["cons_cos_list"])
            pooled[arm]["un"].extend(out["smoothness"]["un_cos_list"])
            by_construction[arm] = out["same_frame_cos_by_construction"]
            for model in c9s["models"]:
                pred_series[arm][model].append(out["predictability"][model]["code"]["r2_vs_copy_last"])

        per_seed.append({
            "seed": seed,
            "split_hash": info["split_hash"],
            "arms": seed_arms,
            "pixel_reference": px,
            "pixel_predictability": pixel_predictability,
        })

    arms_out = {}
    for arm in arms:
        s = series[arm]
        arms_out[arm] = {
            "same_frame_cos_mean": mean_std(s["same_frame_cos"])[0],
            "same_frame_cos_std": mean_std(s["same_frame_cos"])[1],
            "same_frame_cos_by_construction": by_construction[arm],
            "cons_cos_mean": mean_std(s["cons_cos"])[0],
            "cons_cos_std": mean_std(s["cons_cos"])[1],
            "cons_cos_list": pooled[arm]["cons"],
            "un_cos_list": pooled[arm]["un"],
            "recon_mse_mean": mean_std(s["recon_mse"])[0],
            "recon_mse_std": mean_std(s["recon_mse"])[1],
            "energy_mean": mean_std(s["energy"])[0],
            "energy_std": mean_std(s["energy"])[1],
            "tf_mse_mean": mean_std(s["tf_mse"])[0],
            "tf_mse_std": mean_std(s["tf_mse"])[1],
            "copy_last_mse_mean": mean_std(s["copy_last_mse"])[0],
            "mean_frame_mse_mean": mean_std(s["mean_frame_mse"])[0],
            "long_mse_mean": mean_std(s["long_mse"])[0],
            "long_mse_std": mean_std(s["long_mse"])[1],
            "copy_last_long_mse_mean": mean_std(s["copy_last_long_mse"])[0],
            "mean_frame_long_mse_mean": mean_std(s["mean_frame_long_mse"])[0],
            "motion_gap_mean": mean_std(s["motion_gap"])[0],
            "delta_ratio_mean": mean_std(s["delta_ratio"])[0],
            "predictability": {
                model: {
                    "code": {
                        "r2_vs_copy_last_mean": mean_std(pred_series[arm][model])[0],
                        "r2_vs_copy_last_std": mean_std(pred_series[arm][model])[1],
                    }
                }
                for model in c9s["models"]
            },
        }

    pixel_reference_out = {
        "cons_cos_mean": mean_std(pixel_series["cons_cos"])[0],
        "un_cos_mean": mean_std(pixel_series["un_cos"])[0],
        "frac_rel_mean": mean_std(pixel_series["frac_rel"])[0],
        "predictability": {
            model: {
                "r2_vs_copy_last_mean": mean_std(pixel_pred_series[model])[0],
                "r2_vs_copy_last_std": mean_std(pixel_pred_series[model])[1],
            }
            for model in c9s["models"]
        },
    }

    tf_parts = " / ".join(f"{ARM_LABEL[a]}={arms_out[a]['tf_mse_mean']:.4f}" for a in arms)
    long_parts = " / ".join(f"{ARM_LABEL[a]}={arms_out[a]['long_mse_mean']:.4f}" for a in arms)
    cons_parts = " / ".join(f"{ARM_LABEL[a]}={arms_out[a]['cons_cos_mean']:.3f}" for a in arms)
    recon_arms = [a for a in ("iterative", "amortized") if a in arms]
    recon_parts = " / ".join(f"{ARM_LABEL[a]}={arms_out[a]['recon_mse_mean']:.4f}" for a in recon_arms)
    copy_last = arms_out[arms[0]]["copy_last_mse_mean"]
    copy_last_long = arms_out[arms[0]]["copy_last_long_mse_mean"]
    pixel_cons = pixel_reference_out["cons_cos_mean"]
    pred_parts = " / ".join(
        f"{ARM_LABEL[a]}={arms_out[a]['predictability'][primary_model]['code']['r2_vs_copy_last_mean']:.3f}"
        for a in arms
    )
    pixel_r2 = pixel_reference_out["predictability"][primary_model]["r2_vs_copy_last_mean"]

    summary = (
        f"C8 | tf MSE {tf_parts} (copy-last {copy_last:.4f}) | "
        f"long MSE {long_parts} (copy-last {copy_last_long:.4f}) | "
        f"cons cos {cons_parts} (pixel {pixel_cons:.3f}) | "
        f"recon {recon_parts} | "
        f"R² vs copy-last ({primary_model}) {pred_parts} (pixel {pixel_r2:.3f})"
    )
    metrics = {
        "claim": "C8",
        "arms_run": arms,
        "seeds": per_seed,
        "arms": arms_out,
        "pixel_reference": pixel_reference_out,
        "predictability_primary_model": primary_model,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
