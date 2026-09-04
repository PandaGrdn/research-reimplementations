#!/usr/bin/env python3
"""C13 — does test-time settling of the rolled-out code on the inter-layer
consistency term ALONE (no image term — the decoded frame is a deterministic
function of the code, so settling on it is information-free) reduce closed-loop
drift/fade, relative to the pure-latent baseline and to what settling CAN do
when real information (the true frame) is available?

Arms (config `c13.arms`):
  none                    pure latent rollout, r_prev <- r_pred (C12 baseline)
  consistency_top_down    settle_consistency(mode="top_down"), fed back
  consistency_bottom_up   settle_consistency(mode="bottom_up"), fed back
  consistency_joint       settle_consistency(mode="joint"), fed back
  image_settle            settle on the model's OWN decoded frame (C12-style,
                           information-free reference), fed back
  oracle_image            settle warm from r_pred against the TRUE frame
                           (upper reference — real information available)

Each arm runs under both model kinds (`independent`, and `coupled` if
`src.temporal.CoupledTopDownRNN` is available) — for the coupled model the
inter-layer error is exactly zero by construction, so the consistency arms
are asserted no-ops (e1_before < 1e-6): that assertion IS the control.

    python experiments/c13_consistency_rollout.py --smoke
    python experiments/c13_consistency_rollout.py --device mps
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.consistency_settle import consistency_energy, settle_consistency, settle_full
from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    load_temporal_checkpoint,
    parse_args,
    setup,
)
from src.metrics import flat_r, pair_stats, stack_mean_std
from src.offline_gru import encode_eval_codes, train_offline_gru
from src.spatial_pc import f_clamp
from src.utils import clone_r, finish_run, new_run_dir, seed_everything

try:
    from src.offline_gru import cache_eval_codes
except ImportError:
    cache_eval_codes = None

try:
    from src.temporal import CoupledTopDownRNN
except ImportError:
    CoupledTopDownRNN = None

try:
    from src.energy_fade import energy_terms
except ImportError:
    energy_terms = None


DEFAULT_ARMS = [
    "none",
    "consistency_top_down",
    "consistency_bottom_up",
    "consistency_joint",
    "image_settle",
    "oracle_image",
]

ARM_LABEL = {
    "none": "none",
    "consistency_top_down": "cons_td",
    "consistency_bottom_up": "cons_bu",
    "consistency_joint": "cons_joint",
    "image_settle": "image",
    "oracle_image": "oracle",
}

STEP_KEYS = [
    "pixel_mse",
    "cos_r",
    "cos_r_layer1",
    "mass_pred",
    "mass_true",
    "peak_pred",
    "peak_true",
    "r_used_norm",
    "r_true_norm",
    "e1",
    "dc_offset",
    "copy_last_mse",
]


# --------------------------------------------------------------------------
# Defensive wrappers around interfaces owned by the other agents in this
# rework. Each degrades to a local fallback rather than crashing, per SPEC2.
# --------------------------------------------------------------------------

def _get_codes(split_name, seqs, deconvs, r_init, cfg, root, seed, log=print):
    if cache_eval_codes is not None:
        try:
            return cache_eval_codes(split_name, seqs, deconvs, r_init, cfg, root, seed)
        except Exception as e:  # noqa: BLE001 - defensive fallback per SPEC2
            log(f"cache_eval_codes failed ({type(e).__name__}: {e}); falling back to encode_eval_codes")
    return encode_eval_codes(seqs, deconvs, r_init, cfg)


def _load_ckpt_defensive(path, r_init, cfg, device, deconvs=None):
    try:
        return load_temporal_checkpoint(path, r_init, cfg, device, deconvs=deconvs)
    except TypeError:
        return load_temporal_checkpoint(path, r_init, cfg, device)


def _e1_of(I_ref, r, deconvs, alpha, sigma_2, num_layers, use_prior):
    """e1 of an arbitrary code r, preferring src.energy_fade.energy_terms (Agent B)
    and falling back to the local consistency_settle formula (matches SPEC2's
    fallback: 0.5*||r0 - f_clamp(deconv1(r1))||^2)."""
    if energy_terms is not None:
        try:
            terms = energy_terms(I_ref, r, deconvs, alpha, sigma_2, num_layers, use_prior=use_prior, recenter=True)
            return float(terms["e1"])
        except Exception:  # noqa: BLE001 - defensive fallback per SPEC2
            pass
    return consistency_energy(r, deconvs, num_layers)


def _build_model(kind, cfg, r_init, deconvs, device):
    if kind == "independent":
        return build_temporal(cfg, r_init, device)
    t = cfg["temporal"]
    return CoupledTopDownRNN(
        r_init, deconvs, delta_scale=t.get("delta_scale", 1.0), delta_bounded=t.get("delta_bounded", True),
    ).to(device)


def _resolve_model(kind, seed, root, cfg, r_init, deconvs, device, get_train_codes, test_codes, fallback_epochs, fallback_lr, log=print):
    name = f"offline_gru_seed{seed}.pt" if kind == "independent" else f"offline_gru_coupled_seed{seed}.pt"
    ckpt = Path(root) / "artifacts" / name
    if ckpt.exists():
        model = _load_ckpt_defensive(ckpt, r_init, cfg, device, deconvs=deconvs)
        model.to(device).eval()
        log(f"[{kind}] loaded checkpoint {ckpt}")
        return model, True, str(ckpt)
    if kind == "coupled" and CoupledTopDownRNN is None:
        log("[coupled] src.temporal.CoupledTopDownRNN not available yet — skipping the coupled arm.")
        return None, False, None
    log(f"[{kind}] no checkpoint at {ckpt}; training a fallback offline GRU on cached codes (epochs={fallback_epochs})")
    model = _build_model(kind, cfg, r_init, deconvs, device)
    train_codes = get_train_codes()
    train_offline_gru(model, train_codes, test_codes, epochs=fallback_epochs, lr=fallback_lr, device=device, log=log)
    model.eval()
    return model, False, None


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def _prepare_frame(frame):
    x = frame.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x - x.mean()


def _contrast(frame):
    x = frame.detach()
    return float(x.abs().mean().item()), float(x.abs().amax().item())


def _apply_arm(arm, r_pred, deconvs, alpha, lr_r, sigma_2, iters, num_layers, use_prior, I_true_t, coupled):
    if arm == "none":
        return clone_r(r_pred), None
    if arm.startswith("consistency_"):
        mode = arm[len("consistency_"):]
        r_settled, info = settle_consistency(
            r_pred, deconvs, alpha, lr_r, iters, num_layers, mode=mode, use_prior=use_prior
        )
        if coupled:
            assert info["e1_before"] < 1e-6, (
                f"coupled model e1_before={info['e1_before']:.3e} is not ~0 — the "
                "'consistency arms are no-ops under CoupledTopDownRNN' control failed"
            )
        return r_settled, info
    if arm == "image_settle":
        raw = f_clamp(deconvs[0](r_pred[0]))
        I_hat = raw - raw.mean()
        return settle_full(I_hat, r_pred, deconvs, alpha, lr_r, sigma_2, iters, num_layers, use_prior=use_prior)
    if arm == "oracle_image":
        return settle_full(I_true_t, r_pred, deconvs, alpha, lr_r, sigma_2, iters, num_layers, use_prior=use_prior)
    raise ValueError(f"unknown c13 arm '{arm}'")


def _step_record(t, closed_loop, r_used, r_true_t, I_true_t, deconvs, alpha, sigma_2, num_layers, use_prior, copy_last_mse):
    raw_decoded = f_clamp(deconvs[0](r_used[0]))
    dc_offset = float(raw_decoded.mean().item())
    decoded_centered = raw_decoded - raw_decoded.mean()
    pixel_mse = float(torch.mean((decoded_centered - I_true_t) ** 2).item())
    cos_r = pair_stats(r_used, r_true_t)["cos"]
    l1 = min(1, num_layers - 1)
    cos_r_layer1 = pair_stats([r_used[l1]], [r_true_t[l1]])["cos"]
    mass_pred, peak_pred = _contrast(decoded_centered)
    mass_true, peak_true = _contrast(I_true_t)
    e1 = _e1_of(decoded_centered, r_used, deconvs, alpha, sigma_2, num_layers, use_prior)
    return {
        "t": t,
        "closed_loop": bool(closed_loop),
        "pixel_mse": pixel_mse,
        "cos_r": cos_r,
        "cos_r_layer1": cos_r_layer1,
        "mass_pred": mass_pred,
        "mass_true": mass_true,
        "peak_pred": peak_pred,
        "peak_true": peak_true,
        "r_used_norm": float(flat_r(r_used).norm().item()),
        "r_true_norm": float(flat_r(r_true_t).norm().item()),
        "e1": e1,
        "dc_offset": dc_offset,
        "copy_last_mse": copy_last_mse,
    }


def _teacher_forced_prefix(codes_seq, model, split_point):
    """Teacher-force t=0..split_point-1 once; hand off (hidden, r_prev) shared
    by every arm so arms differ only after the handoff, per SPEC2."""
    hidden = model.init_hidden(codes_seq[0])
    r_prev = None
    with torch.no_grad():
        for t in range(split_point):
            r_in = [torch.zeros_like(ri) for ri in codes_seq[0]] if r_prev is None else r_prev
            _, hidden, _ = model(r_in, hidden)
            hidden = [hi.detach() for hi in hidden]
            r_prev = clone_r(codes_seq[t])
    return hidden, r_prev


def rollout_sequence(seq, codes, model, coupled, deconvs, cfg, split_point, arms, iters, lr_r):
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    alpha, sigma_2 = inf["alpha"], inf["sigma_2"]
    use_prior = inf.get("use_prior", True)

    seq_ = seq.float()
    if seq_.ndim == 3:
        seq_ = seq_.unsqueeze(1)
    T = seq_.shape[0]
    sp = min(split_point, T - 1) if T > 1 else 0
    I_true = [_prepare_frame(seq_[t]) for t in range(T)]
    copy_last_frozen = I_true[max(sp - 1, 0)]

    hidden0, r_prev0 = _teacher_forced_prefix(codes, model, sp)
    pre_steps = []
    for t in range(1, sp):
        copy_mse = float(torch.mean((copy_last_frozen - I_true[t]) ** 2).item())
        pre_steps.append(
            _step_record(t, False, codes[t], codes[t], I_true[t], deconvs, alpha, sigma_2, num_layers, use_prior, copy_mse)
        )

    out = {}
    for arm in arms:
        hidden = [hi.clone() for hi in hidden0]
        r_prev = clone_r(r_prev0)
        steps = list(pre_steps)
        with torch.no_grad():
            for t in range(sp, T):
                r_pred, hidden, _ = model(r_prev, hidden)
                r_used, info = _apply_arm(
                    arm, r_pred, deconvs, alpha, lr_r, sigma_2, iters, num_layers, use_prior, I_true[t], coupled
                )
                copy_mse = float(torch.mean((copy_last_frozen - I_true[t]) ** 2).item())
                rec = _step_record(
                    t, True, r_used, codes[t], I_true[t], deconvs, alpha, sigma_2, num_layers, use_prior, copy_mse
                )
                if info is not None and "e1_before" in info:
                    rec["e1_before"] = info["e1_before"]
                    rec["e1_after"] = info["e1_after"]
                steps.append(rec)
                hidden = [hi.detach() for hi in hidden]
                r_prev = clone_r(r_used)
        out[arm] = steps
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _seq_post_mean(steps, key):
    vals = [s[key] for s in steps if s["closed_loop"]]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def aggregate_arm(per_seq_steps):
    if not per_seq_steps:
        return {"curves": {}, "pre_split": {}, "post_split": {}, "per_seq_pixel_mse": [], "per_seq_cos_r": [], "n_sequences": 0}
    curves = {}
    for k in STEP_KEYS:
        mu, sd = stack_mean_std([[s[k] for s in steps] for steps in per_seq_steps])
        curves[f"{k}_mean"] = mu
        curves[f"{k}_std"] = sd

    def _side(pred):
        vals = {k: [] for k in STEP_KEYS}
        for steps in per_seq_steps:
            for s in steps:
                if pred(s):
                    for k in STEP_KEYS:
                        vals[k].append(s[k])
        out = {}
        for k, xs in vals.items():
            arr = torch.tensor(xs, dtype=torch.float64) if xs else torch.tensor([float("nan")])
            out[k] = {
                "mean": float(arr.mean().item()) if xs else float("nan"),
                "std": float(arr.std(unbiased=False).item()) if len(xs) > 1 else 0.0,
                "n": len(xs),
            }
        return out

    pre = _side(lambda s: not s["closed_loop"])
    post = _side(lambda s: s["closed_loop"])
    return {
        "curves": curves,
        "pre_split": pre,
        "post_split": post,
        "per_seq_pixel_mse": [_seq_post_mean(steps, "pixel_mse") for steps in per_seq_steps],
        "per_seq_cos_r": [_seq_post_mean(steps, "cos_r") for steps in per_seq_steps],
        "n_sequences": len(per_seq_steps),
    }


def _annotate_deltas(arm_agg, arms):
    if "none" not in arm_agg:
        return
    none = arm_agg["none"]
    none_pixel_mean = none["post_split"].get("pixel_mse", {}).get("mean")
    none_cos_mean = none["post_split"].get("cos_r", {}).get("mean")
    none_pixel_list = none["per_seq_pixel_mse"]
    for arm in arms:
        if arm not in arm_agg or arm == "none":
            continue
        a = arm_agg[arm]
        if none_pixel_mean is not None and "pixel_mse" in a["post_split"]:
            a["post_split"]["pixel_mse"]["delta_vs_none"] = a["post_split"]["pixel_mse"]["mean"] - none_pixel_mean
        if none_cos_mean is not None and "cos_r" in a["post_split"]:
            a["post_split"]["cos_r"]["delta_vs_none"] = a["post_split"]["cos_r"]["mean"] - none_cos_mean
        beat = sum(
            1 for x, y in zip(a["per_seq_pixel_mse"], none_pixel_list) if x == x and y == y and x < y
        )  # x == x filters NaN
        a["beat_none_count"] = beat
        a["beat_none_total"] = len(a["per_seq_pixel_mse"])
    arm_agg["none"]["post_split"].setdefault("pixel_mse", {})["delta_vs_none"] = 0.0
    arm_agg["none"]["post_split"].setdefault("cos_r", {})["delta_vs_none"] = 0.0


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "nan"


def build_summary(models_out, arms):
    indep = (models_out.get("independent") or {}).get("arms", {})
    mse_parts = [
        f"{ARM_LABEL.get(a, a)}={_fmt(indep[a]['post_split']['pixel_mse']['mean'])}"
        for a in arms
        if a in indep
    ]
    cos_parts = [
        f"{ARM_LABEL.get(a, a)}={_fmt(indep[a]['post_split']['cos_r']['mean'], 3)}"
        for a in arms
        if a in indep
    ]
    copy_last = indep.get("none", {}).get("post_split", {}).get("copy_last_mse", {}).get("mean")
    beat_str = ""
    if "consistency_top_down" in indep:
        b = indep["consistency_top_down"]
        beat_str = f" | cons_td beat none on {b.get('beat_none_count', 0)}/{b.get('beat_none_total', 0)} seqs"
    coupled_str = ""
    coupled = (models_out.get("coupled") or {}).get("arms", {})
    if coupled:
        cn = coupled.get("none", {}).get("post_split", {}).get("pixel_mse", {}).get("mean")
        co = coupled.get("oracle_image", {}).get("post_split", {}).get("pixel_mse", {}).get("mean")
        coupled_str = f" | coupled: none={_fmt(cn)} oracle={_fmt(co)} (e1≡0)"
    return (
        f"C13 | indep post-split MSE: {'  '.join(mse_parts)} (copy-last {_fmt(copy_last)}) | "
        f"post cos: {'  '.join(cos_parts)}{beat_str}{coupled_str}"
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args("C13 consistency-only test-time settle on closed-loop rollout")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c13_consistency_rollout", arm=arm_name(cfg))

    c13 = cfg.get("c13", {})
    c11cfg = cfg.get("c11", {})
    headline_sp = int(cfg["eval"].get("headline_split", 10))
    n_seq_want = c13.get("n_sequences", 32)
    iters = c13.get("iters", 50)
    lr_r = c13.get("lr_r") or cfg["inference"]["lr_r"]
    arms = c13.get("arms", DEFAULT_ARMS)
    fallback_epochs = c13.get("fallback_epochs", 5)
    fallback_lr = c11cfg.get("lr", 1e-3)

    kinds = ["independent"]
    if CoupledTopDownRNN is not None:
        kinds.append("coupled")
    else:
        print("CoupledTopDownRNN unavailable (src.temporal has not landed it yet) — skipping the coupled model kind.")

    pooled_steps = {kind: {arm: [] for arm in arms} for kind in kinds}
    per_seed_records = []
    ckpt_info = {kind: [] for kind in kinds}

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        n_use = min(n_seq_want, len(test))
        n_test_for_fallback = min(len(test), c11cfg.get("n_test_sequences", cfg["data"]["n_test"]))
        n_test_codes = max(n_use, n_test_for_fallback)
        test_seqs_full = [s.to(device) for s in test[:n_test_codes]]
        test_codes_full = _get_codes("test", test_seqs_full, deconvs, r_init, cfg, root, seed, log=print)
        eval_seqs = test_seqs_full[:n_use]
        eval_codes = test_codes_full[:n_use]
        test_codes_for_fallback = test_codes_full[:n_test_for_fallback]

        train_codes_cache = {}

        def _get_train_codes():
            if "codes" not in train_codes_cache:
                n_train_seq = c11cfg.get("n_train_sequences", cfg["data"]["n_train"])
                seqs = [s.to(device) for s in train[:n_train_seq]]
                train_codes_cache["codes"] = _get_codes("train", seqs, deconvs, r_init, cfg, root, seed, log=print)
            return train_codes_cache["codes"]

        seed_record = {"seed": seed, "kinds": {}}
        for kind in kinds:
            model, loaded, ckpt_path = _resolve_model(
                kind, seed, root, cfg, r_init, deconvs, device,
                _get_train_codes, test_codes_for_fallback, fallback_epochs, fallback_lr, log=print,
            )
            if model is None:
                continue
            ckpt_info[kind].append({"seed": seed, "loaded": loaded, "path": ckpt_path})

            per_arm_steps = {arm: [] for arm in arms}
            for seq, codes in zip(eval_seqs, eval_codes):
                out = rollout_sequence(seq, codes, model, kind == "coupled", deconvs, cfg, headline_sp, arms, iters, lr_r)
                for arm in arms:
                    per_arm_steps[arm].append(out[arm])
                    pooled_steps[kind][arm].append(out[arm])

            arm_agg = {arm: aggregate_arm(per_arm_steps[arm]) for arm in arms}
            _annotate_deltas(arm_agg, arms)
            seed_record["kinds"][kind] = {"arms": arm_agg, "n_sequences": n_use}
            none_post = arm_agg.get("none", {}).get("post_split", {}).get("pixel_mse", {}).get("mean")
            print(f"C13 | seed={seed} kind={kind}  n={n_use}  none post-split MSE={_fmt(none_post)}")

        per_seed_records.append(seed_record)

    models_out = {}
    for kind in kinds:
        n_pooled = len(pooled_steps[kind].get(arms[0], [])) if arms else 0
        if n_pooled == 0:
            continue
        arm_agg = {arm: aggregate_arm(pooled_steps[kind][arm]) for arm in arms}
        _annotate_deltas(arm_agg, arms)
        models_out[kind] = {"arms": arm_agg, "n_sequences": n_pooled}

    summary = build_summary(models_out, arms)
    metrics = {
        "claim": "C13",
        "headline_split": headline_sp,
        "iters": iters,
        "lr_r": lr_r,
        "arms": arms,
        "n_sequences": n_seq_want,
        "seeds": per_seed_records,
        "models": models_out,
        "checkpoints": ckpt_info,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
