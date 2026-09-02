"""Pair stats, copy-last metrics, noise-floor decomposition."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def flat_r(r_list):
    if torch.is_tensor(r_list):
        return r_list.reshape(-1)
    return torch.cat([ri.reshape(-1) for ri in r_list])


def pair_stats(ra, rb):
    a, b = flat_r(ra), flat_r(rb)
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    rel = (torch.norm(a - b) / (torch.norm(a) + 1e-8)).item()
    abs_dist = torch.norm(a - b).item()
    return {"cos": cos, "rel": rel, "abs": abs_dist}


_pair_stats = pair_stats
_flat_r = flat_r


def summarize_pair_lists(cos_list, rel_list, abs_list):
    return {
        "cos": float(np.mean(cos_list)) if cos_list else float("nan"),
        "rel": float(np.mean(rel_list)) if rel_list else float("nan"),
        "abs": float(np.mean(abs_list)) if abs_list else float("nan"),
        "cos_std": float(np.std(cos_list)) if cos_list else float("nan"),
        "rel_std": float(np.std(rel_list)) if rel_list else float("nan"),
        "abs_std": float(np.std(abs_list)) if abs_list else float("nan"),
    }


def motion_gap(mse_vs_curr, mse_vs_prev):
    mean_curr = sum(mse_vs_curr) / max(len(mse_vs_curr), 1)
    mean_prev = sum(mse_vs_prev) / max(len(mse_vs_prev), 1)
    return mean_curr - mean_prev, mean_curr, mean_prev


def delta_ratio(delta_norms, dr_norms):
    mean_delta = sum(delta_norms) / max(len(delta_norms), 1)
    mean_dr = sum(dr_norms) / max(len(dr_norms), 1)
    return mean_delta / (mean_dr + 1e-8), mean_delta, mean_dr


def saturation_frac(frame, thresh=0.99):
    x = frame.detach()
    return (x.abs() > thresh).float().mean().item()


def per_frame_mse(true_frames, pred_frames):
    err = (true_frames - pred_frames) ** 2
    while err.ndim > 1:
        err = err.mean(dim=-1)
    return err


def long_horizon_mean(mse_per_frame, split_point):
    """Mean MSE from the first predicted frame at/after split through the end.

    mse_per_frame[k] is the prediction for sequence frame t=k+1.
    split_point indexes the raw sequence, so the handoff lives at index split_point-1.
    """
    curve = list(mse_per_frame)
    start = max(split_point - 1, 0)
    sl = curve[start:]
    if not sl:
        return float("nan")
    return float(sum(sl) / len(sl))


def noise_floor_decomposition(settle_abs, target_abs):
    """Centerpiece: 2 σ_settle² vs ||r_t − r_{t−1}||².

    Two independent settles of the same frame differ by ||ε1−ε2|| = settle_abs,
    so per-inference noise σ = settle_abs / √2 and the noise energy in a
    consecutive difference is 2σ² = settle_abs².
    """
    sigma = settle_abs / math.sqrt(2.0)
    noise_energy = 2.0 * (sigma ** 2)
    target_energy = target_abs ** 2
    noise_share = float(min(1.0, noise_energy / (target_energy + 1e-12)))
    predictable_fraction = float(max(0.0, 1.0 - noise_share))
    return {
        "settle_abs": float(settle_abs),
        "sigma_settle": float(sigma),
        "target_abs": float(target_abs),
        "noise_energy": float(noise_energy),
        "target_energy": float(target_energy),
        "noise_share": noise_share,
        "signal_share": predictable_fraction,
        "predictable_fraction": predictable_fraction,
    }


def stack_mean_std(curves):
    """curves: list of equal-length lists → mean/std per index."""
    arr = np.asarray(curves, dtype=np.float64)
    if arr.size == 0:
        return [], []
    if arr.ndim == 1:
        return arr.tolist(), [0.0] * len(arr)
    return arr.mean(axis=0).tolist(), arr.std(axis=0, ddof=0).tolist()
