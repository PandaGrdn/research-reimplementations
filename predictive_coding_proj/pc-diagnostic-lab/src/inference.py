"""Iterative settle inference, ported from temporal_predictive_coding.ipynb."""

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import pair_stats, summarize_pair_lists
from src.spatial_pc import f_clamp


def _prepare_image(I):
    I_curr = I.float()
    if I_curr.ndim == 3:
        I_curr = I_curr.unsqueeze(0)
    return I_curr - I_curr.mean()


def settle_grounded(
    I,
    r_init,
    deconvs,
    alpha,
    lr_r,
    sigma_2,
    num_epochs_inner,
    num_layers,
    r_warm=None,
    conv_tol=None,
    init_noise=0.01,
    use_prior=True,
):
    """Encode one frame with spatial settle only (no temporal prior).

    r_warm: if given, start from these activations (e.g. previous frame).
    Otherwise start from r_init shapes with fresh Gaussian noise (non-determinism test).
    conv_tol: if set, stop early when mean ||dr|| < conv_tol.
    """
    I_curr = _prepare_image(I)

    if r_warm is not None:
        r_curr = [ri.detach().clone() for ri in r_warm]
    else:
        r_curr = [torch.randn_like(ri) * init_noise for ri in r_init]

    e_spatial = [None] * num_layers
    n_used = num_epochs_inner
    for it in range(num_epochs_inner):
        e_spatial[0] = I_curr - f_clamp(deconvs[0](r_curr[0]))
        for i in range(1, num_layers):
            e_spatial[i] = r_curr[i - 1] - f_clamp(deconvs[i](r_curr[i]))
        max_step = 0.0
        for i in range(num_layers):
            cauchy_prior = alpha * (2 * r_curr[i] / (1 + r_curr[i] ** 2)) if use_prior else 0.0
            bottom_up = F.conv2d(
                e_spatial[i],
                deconvs[i].weight,
                padding=deconvs[i].padding,
                stride=deconvs[i].stride,
            )
            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior
            if i < num_layers - 1:
                dr = dr - (1.0 / sigma_2) * e_spatial[i + 1]
            r_curr[i] = (r_curr[i] + lr_r * dr).detach()
            max_step = max(max_step, torch.mean(dr.abs()).item())
        if conv_tol is not None and max_step < conv_tol:
            n_used = it + 1
            break
    return [ri.detach().clone() for ri in r_curr], n_used


def settle_with_temporal_prior(
    I,
    r_curr,
    r_pred,
    deconvs,
    alpha,
    lr_r,
    sigma_2,
    num_epochs_inner,
    num_layers,
    r_prev1=None,
    lambda_slow=0.0,
    use_prior=True,
):
    """Inner settle used inside train_loop / teacher-forced validation."""
    I_curr = _prepare_image(I)
    e_spatial = [None] * num_layers
    has_prev = r_prev1 is not None
    for _ in range(num_epochs_inner):
        e_spatial[0] = I_curr - f_clamp(deconvs[0](r_curr[0]))
        for i in range(1, num_layers):
            e_spatial[i] = r_curr[i - 1] - f_clamp(deconvs[i](r_curr[i]))
        for i in range(num_layers):
            cauchy_prior = alpha * (2 * r_curr[i] / (1 + r_curr[i] ** 2)) if use_prior else 0.0
            bottom_up = F.conv2d(
                e_spatial[i],
                deconvs[i].weight,
                padding=deconvs[i].padding,
                stride=deconvs[i].stride,
            )
            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior - (1.0 / (sigma_2 * 100)) * (
                r_curr[i] - r_pred[i].detach()
            )
            if has_prev and lambda_slow > 0:
                dr = dr - 2.0 * lambda_slow * (r_curr[i] - r_prev1[i])
            if i < num_layers - 1:
                dr = dr - (1.0 / sigma_2) * e_spatial[i + 1]
            r_curr[i] = (r_curr[i] + lr_r * dr).detach()
            r_curr[i].requires_grad = True
    return r_curr, e_spatial, I_curr


def diagnose_inference_nondeterminism(
    frames,
    r_init,
    deconvs,
    alpha=0.001,
    lr_r=0.005,
    sigma_2=1.0,
    num_epochs_inner=50,
    num_layers=2,
    n_frames=5,
    init_noise=0.01,
    use_prior=True,
    log=print,
):
    """Settle the same frame twice from different random inits; report cos."""
    log("\n" + "=" * 60)
    log(f"A. Inference non-determinism  (same frame ×2 random inits, inner={num_epochs_inner}, lr_r={lr_r})")
    log("=" * 60)
    cos_list, rel_list, abs_list = [], [], []
    for k, I in enumerate(frames[:n_frames]):
        r1, n1 = settle_grounded(
            I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            init_noise=init_noise, use_prior=use_prior,
        )
        r2, n2 = settle_grounded(
            I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            init_noise=init_noise, use_prior=use_prior,
        )
        stats = pair_stats(r1, r2)
        cos_list.append(stats["cos"])
        rel_list.append(stats["rel"])
        abs_list.append(stats["abs"])
        log(f"  frame {k}: cos={stats['cos']:.4f}  rel={stats['rel']:.4f}  ||dr||={stats['abs']:.4f}  iters={n1}/{n2}")
    out = summarize_pair_lists(cos_list, rel_list, abs_list)
    log(f"  MEAN cos={out['cos']:.4f}  rel={out['rel']:.4f}")
    if out["cos"] < 0.7:
        log("  VERDICT: inference does NOT converge to a unique code — 'chaos' is largely settle noise.")
    else:
        log("  VERDICT: settle is fairly deterministic — remaining jumpiness is more about the code.")
    out.update({"cos_list": cos_list, "rel_list": rel_list, "abs_list": abs_list, "iters": num_epochs_inner, "lr_r": lr_r})
    return out


def diagnose_latent_smoothness(
    seq,
    unrelated_frames,
    r_init,
    deconvs,
    alpha=0.001,
    lr_r=0.005,
    sigma_2=1.0,
    num_epochs_inner=50,
    num_layers=2,
    max_unrelated_pairs=40,
    warm_start=False,
    label="",
    init_noise=0.01,
    use_prior=True,
    generator=None,
    log=print,
):
    """Grounded trajectory vs unrelated-frame distances in r-space."""
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]
    tag = label or ("warm-start" if warm_start else "cold-start")

    log("\n" + "=" * 60)
    log(f"B. Latent smoothness [{tag}]  inner={num_epochs_inner}  lr_r={lr_r}")
    log("=" * 60)

    log("Settling grounded trajectory...")
    rs = []
    r_prev = None
    for t in range(T):
        r_t, _ = settle_grounded(
            seq[t], r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            r_warm=r_prev if warm_start else None,
            init_noise=init_noise,
            use_prior=use_prior,
        )
        rs.append(r_t)
        r_prev = r_t

    cons_cos, cons_rel, cons_abs = [], [], []
    log(f"\n{'t':>4}  {'cos(r_t,r_t-1)':>14}  {'||dr||/||r_t||':>14}  {'||dr||':>10}")
    for t in range(1, T):
        stats = pair_stats(rs[t], rs[t - 1])
        cons_cos.append(stats["cos"])
        cons_rel.append(stats["rel"])
        cons_abs.append(stats["abs"])
        log(f"{t:4d}  {stats['cos']:14.4f}  {stats['rel']:14.4f}  {stats['abs']:10.4f}")

    log("\nSettling unrelated frames (always cold-start)...")
    u_rs = []
    for I in unrelated_frames:
        r_u, _ = settle_grounded(
            I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            init_noise=init_noise, use_prior=use_prior,
        )
        u_rs.append(r_u)
    un_cos, un_rel, un_abs = [], [], []
    n_u = len(u_rs)
    pairs = [(i, j) for i in range(n_u) for j in range(i + 1, n_u)]
    if len(pairs) > max_unrelated_pairs:
        g = generator if generator is not None else torch.Generator(device="cpu")
        idx = torch.randperm(len(pairs), generator=g)[:max_unrelated_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        stats = pair_stats(u_rs[i], u_rs[j])
        un_cos.append(stats["cos"])
        un_rel.append(stats["rel"])
        un_abs.append(stats["abs"])

    m_cc, m_cr, m_ca = float(np.mean(cons_cos)), float(np.mean(cons_rel)), float(np.mean(cons_abs))
    m_uc, m_ur, m_ua = float(np.mean(un_cos)), float(np.mean(un_rel)), float(np.mean(un_abs))
    frac_rel = m_cr / (m_ur + 1e-8)
    frac_abs = m_ca / (m_ua + 1e-8)

    log(f"\nConsecutive ({tag}):  cos={m_cc:.4f}  rel={m_cr:.4f}  ||dr||={m_ca:.4f}")
    log(f"Unrelated frames:      cos={m_uc:.4f}  rel={m_ur:.4f}  ||dr||={m_ua:.4f}")
    log(f"consec/unrelated:      rel={frac_rel:.3f}  abs={frac_abs:.3f}")
    if frac_rel > 0.7:
        log("VERDICT: NON-SMOOTH — consecutive ≈ unrelated.")
    elif frac_rel > 0.4:
        log("VERDICT: moderately jumpy.")
    else:
        log("VERDICT: usable — consecutive clearly closer than unrelated.")
    return {
        "cons_cos": m_cc,
        "cons_rel": m_cr,
        "cons_abs": m_ca,
        "un_cos": m_uc,
        "un_rel": m_ur,
        "un_abs": m_ua,
        "frac_rel": frac_rel,
        "frac_abs": frac_abs,
        "tag": tag,
        "n_cons": len(cons_cos),
        "n_un": len(un_cos),
        "cons_cos_list": cons_cos,
        "un_cos_list": un_cos,
        "cons_abs_list": cons_abs,
        "un_abs_list": un_abs,
    }


def collect_unrelated_frames(seqs, n_frames, seed=0):
    frames = []
    for seq in seqs:
        for t in range(seq.shape[0]):
            frames.append(seq[t])
            if len(frames) >= n_frames:
                return frames
    return frames


def collect_eval_frames(seqs, n_frames):
    frames = []
    for seq in seqs:
        for t in range(seq.shape[0]):
            frames.append(seq[t])
            if len(frames) >= n_frames:
                return frames
    return frames
