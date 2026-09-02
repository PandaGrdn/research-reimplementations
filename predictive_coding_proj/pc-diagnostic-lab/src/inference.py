"""Iterative settle inference, ported from temporal_predictive_coding.ipynb."""

import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import flat_r, pair_stats, summarize_pair_lists
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


def settle_info(I, r, deconvs, alpha, sigma_2, num_layers, use_prior=True):
    """PC energy at r plus the mean |dr| the next settle_grounded step would take.

    Does not mutate r or apply any update; purely diagnostic (no_grad).
    """
    with torch.no_grad():
        I_curr = _prepare_image(I)
        r = [ri.detach().clone() for ri in r]
        e_spatial = [None] * num_layers
        e_spatial[0] = I_curr - f_clamp(deconvs[0](r[0]))
        for i in range(1, num_layers):
            e_spatial[i] = r[i - 1] - f_clamp(deconvs[i](r[i]))

        energy = [float(torch.sum(e_spatial[i] ** 2).item() / (2 * sigma_2)) for i in range(num_layers)]
        prior_energy = 0.0
        if use_prior:
            for i in range(num_layers):
                prior_energy += alpha * float(torch.sum(torch.log(1 + r[i] ** 2)).item())
        total_energy = float(sum(energy) + prior_energy)

        dr_abs = []
        for i in range(num_layers):
            cauchy_prior = alpha * (2 * r[i] / (1 + r[i] ** 2)) if use_prior else 0.0
            bottom_up = F.conv2d(
                e_spatial[i],
                deconvs[i].weight,
                padding=deconvs[i].padding,
                stride=deconvs[i].stride,
            )
            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior
            if i < num_layers - 1:
                dr = dr - (1.0 / sigma_2) * e_spatial[i + 1]
            dr_abs.append(dr.abs().reshape(-1))
        mean_abs_dr = float(torch.cat(dr_abs).mean().item())

    return {
        "energy": energy,
        "prior_energy": float(prior_energy),
        "total_energy": total_energy,
        "mean_abs_dr": mean_abs_dr,
    }


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
    temporal_prior_weight=0.01,
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
            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior - (temporal_prior_weight / sigma_2) * (
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
    """Settle the same frame twice from different random inits; report cos.

    Also records, per frame, the PC energy / mean |dr| at each of the two settled
    codes (via settle_info) and the settled ||r||, so callers can tell apart
    "irreducible noise" (energy still high, settle not converged) from a converged
    null-space discrepancy (energy ~0, codes still differ because the init_noise
    seed was never driven out of the null space of an overcomplete dictionary).
    """
    log("\n" + "=" * 60)
    log(f"A. Inference non-determinism  (same frame ×2 random inits, inner={num_epochs_inner}, lr_r={lr_r})")
    log("=" * 60)
    cos_list, rel_list, abs_list = [], [], []
    energy_list, mean_abs_dr_list, r_norm_list = [], [], []
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
        info1 = settle_info(I, r1, deconvs, alpha, sigma_2, num_layers, use_prior=use_prior)
        info2 = settle_info(I, r2, deconvs, alpha, sigma_2, num_layers, use_prior=use_prior)
        energy_list.append(0.5 * (info1["total_energy"] + info2["total_energy"]))
        mean_abs_dr_list.append(0.5 * (info1["mean_abs_dr"] + info2["mean_abs_dr"]))
        r_norm_list.append(0.5 * (flat_r(r1).norm().item() + flat_r(r2).norm().item()))
        log(
            f"  frame {k}: cos={stats['cos']:.4f}  rel={stats['rel']:.4f}  ||dr||={stats['abs']:.4f}  "
            f"energy={energy_list[-1]:.4f}  ||r||={r_norm_list[-1]:.4f}  iters={n1}/{n2}"
        )
    out = summarize_pair_lists(cos_list, rel_list, abs_list)
    out["energy_mean"] = float(np.mean(energy_list)) if energy_list else float("nan")
    out["mean_abs_dr_mean"] = float(np.mean(mean_abs_dr_list)) if mean_abs_dr_list else float("nan")
    out["r_norm_mean"] = float(np.mean(r_norm_list)) if r_norm_list else float("nan")
    log(f"  MEAN cos={out['cos']:.4f}  rel={out['rel']:.4f}  energy={out['energy_mean']:.4f}  ||r||={out['r_norm_mean']:.4f}")
    if out["cos"] < 0.7:
        log("  VERDICT: inference does NOT converge to a unique code — 'chaos' is largely settle noise.")
    else:
        log("  VERDICT: settle is fairly deterministic — remaining jumpiness is more about the code.")
    out.update({
        "cos_list": cos_list,
        "rel_list": rel_list,
        "abs_list": abs_list,
        "energy_list": energy_list,
        "mean_abs_dr_list": mean_abs_dr_list,
        "r_norm_list": r_norm_list,
        "iters": num_epochs_inner,
        "lr_r": lr_r,
    })
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
    unrelated_pairs=None,
    log=print,
):
    """Grounded trajectory vs unrelated-frame distances in r-space.

    unrelated_pairs: if given, an explicit list of (i, j) index pairs into
    unrelated_frames to use instead of the internal subsample — lets callers
    (e.g. C5's cold/warm/warm_zero_init/pixel conditions) reuse the exact same
    seeded pair selection across conditions for a paired comparison.
    """
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
    if unrelated_pairs is not None:
        pairs = unrelated_pairs
    else:
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
        "cons_rel_list": cons_rel,
        "un_rel_list": un_rel,
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


def settle_trajectory(
    seq,
    r_init,
    deconvs,
    alpha,
    lr_r,
    sigma_2,
    num_epochs_inner,
    num_layers,
    init_noise=0.01,
    use_prior=True,
    fixed_start=False,
):
    """Warm-started settle over a whole sequence; returns the per-frame code list.

    fixed_start=False: frame 0 starts cold from torch.randn(...) * init_noise (a
    fresh random init — independent across separate calls, since settle_grounded
    draws from the global RNG). fixed_start=True: frame 0 starts from the
    checkpointed r_init itself (r_warm=r_init), matching what the real
    train/eval pipeline does — deterministic, so two independent calls with
    fixed_start=True produce identical trajectories.

    Used by C6 to build two independent rollouts (protocols "warm_independent_init"
    and "pipeline") without depending on diagnose_latent_smoothness's internal
    aggregation.
    """
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]
    rs = []
    r_prev = None
    for t in range(T):
        r_warm = r_init if (t == 0 and fixed_start) else r_prev
        r_t, _ = settle_grounded(
            seq[t], r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            r_warm=r_warm, init_noise=init_noise, use_prior=use_prior,
        )
        rs.append(r_t)
        r_prev = r_t
    return rs
