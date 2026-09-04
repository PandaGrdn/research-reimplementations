"""C13 — consistency-only test-time settle: gradient descent on the inter-layer
term ALONE, with no image term anywhere (the decoded frame is a deterministic
function of the code, so settling on it carries zero information; only the
hierarchical-consistency residual and the priors do).

E_c = (1/2) * sum_{i=1..num_layers-1} ||r[i-1] - f_clamp(deconvs[i](r[i]))||^2 + prior

Update form is lifted directly from `settle_grounded` in src/inference.py so the
per-iteration step sizes are directly comparable to the C12-style full settle:
bottom-up conv2d (adjoint of the ConvTranspose2d decoder) drives the "upper"
member of each inter-layer pair, the residual itself (with a minus sign) drives
the "lower" member — just with the image term (layer 0's own reconstruction
error, e_spatial[0] in settle_grounded) dropped everywhere, and no 1/sigma_2
scaling (the E_c defined above has no sigma_2 in it).
"""

import torch
import torch.nn.functional as F

from src.inference import settle_grounded
from src.spatial_pc import f_clamp


def _inter_layer_errors(r, deconvs, num_layers):
    """e[i] = r[i-1] - f_clamp(deconvs[i](r[i])) for i=1..num_layers-1; e[0] unused (no image term)."""
    e = [None] * num_layers
    for i in range(1, num_layers):
        e[i] = r[i - 1] - f_clamp(deconvs[i](r[i]))
    return e


def consistency_energy(r, deconvs, num_layers):
    """0.5 * sum_i ||e[i]||^2 over all inter-layer residuals — the E_c reconstruction
    term (no prior). Public so callers (e.g. the C13 experiment script) can compute
    "e1" of an arbitrary code without going through a settle."""
    e = _inter_layer_errors(r, deconvs, num_layers)
    total = 0.0
    for ei in e:
        if ei is not None:
            total += 0.5 * float(torch.sum(ei ** 2).item())
    return total


def _update_mask(num_layers, mode):
    """Which layer indices get updated for each mode. For num_layers=2 (the only
    configuration this lab actually runs), this is exactly the spec's definition:
    top_down updates r0 only (r1 fixed), bottom_up updates r1 only (r0 fixed),
    joint updates both. Generalized to num_layers>2 as "all but the top" /
    "only the top" / "everything" — untested territory since num_layers is always
    2 in configs/base.yaml and configs/smoke.yaml, but keeps the function total."""
    if mode == "top_down":
        return [True] * (num_layers - 1) + [False]
    if mode == "bottom_up":
        return [False] * (num_layers - 1) + [True]
    if mode == "joint":
        return [True] * num_layers
    raise ValueError(f"unknown mode '{mode}' (expected 'top_down', 'bottom_up', or 'joint')")


def settle_consistency(r, deconvs, alpha, lr_r, iters, num_layers, mode, use_prior=True):
    """Gradient descent on E_c (inter-layer consistency only, NO image term).

    r: list of per-layer codes (r[0] = bottom/largest layer ... r[num_layers-1] = top).
    mode: "top_down" | "bottom_up" | "joint" (see _update_mask).

    Per-iteration update (mirrors settle_grounded's dr formula, minus the image
    term and the 1/sigma_2 scale, per E_c's definition above):
      dr[i] += conv2d(e[i], deconvs[i].weight, padding=.., stride=..)   if i > 0   (bottom-up)
      dr[i] -= e[i+1]                                                   if i < L-1 (top-down)
      dr[i] -= cauchy_prior(r[i])                                       if use_prior
    then r[i] <- r[i] + lr_r * dr[i], only for i where the mode's mask is True.

    Returns (r_new, info) with info = {"e1_before", "e1_after", "iters"}, e1_* being
    consistency_energy(r, ...) before/after the settle (0.5 * sum ||e_i||^2, summed
    over every inter-layer pair — for num_layers=2 this is exactly the spec's
    "e1" = 0.5*||r0 - f_clamp(deconv1(r1))||^2).
    """
    if num_layers < 2:
        raise ValueError("settle_consistency needs num_layers >= 2 (no inter-layer pair otherwise)")
    mask = _update_mask(num_layers, mode)
    r_curr = [ri.detach().clone() for ri in r]

    e1_before = consistency_energy(r_curr, deconvs, num_layers)

    for _ in range(iters):
        e = _inter_layer_errors(r_curr, deconvs, num_layers)
        for i in range(num_layers):
            if not mask[i]:
                continue
            cauchy_prior = alpha * (2 * r_curr[i] / (1 + r_curr[i] ** 2)) if use_prior else 0.0
            dr = -cauchy_prior
            if i > 0:
                dr = dr + F.conv2d(
                    e[i], deconvs[i].weight, padding=deconvs[i].padding, stride=deconvs[i].stride,
                )
            if i < num_layers - 1:
                dr = dr - e[i + 1]
            r_curr[i] = (r_curr[i] + lr_r * dr).detach()

    e1_after = consistency_energy(r_curr, deconvs, num_layers)

    return r_curr, {"e1_before": e1_before, "e1_after": e1_after, "iters": iters}


def settle_full(I_hat, r, deconvs, alpha, lr_r, sigma_2, iters, num_layers, use_prior=True):
    """Reference (information-carrying-image) settle: the C12-style image+consistency
    settle, warm-started from the rolled-out code `r`. Thin wrapper around
    settle_grounded so the image_settle / oracle_image arms of C13 go through the
    same settle machinery as the consistency-only arms — one code path away.

    I_hat: the frame to settle against (mean-centred already or not — settle_grounded
    re-centres internally). For `image_settle` this is the model's own decoded frame
    (information-free: a function of r); for `oracle_image` it is the true frame at
    this timestep (the upper reference — real information is available).
    """
    r_new, n_used = settle_grounded(
        I_hat,
        r_init=r,
        deconvs=deconvs,
        alpha=alpha,
        lr_r=lr_r,
        sigma_2=sigma_2,
        num_epochs_inner=iters,
        num_layers=num_layers,
        r_warm=r,
        use_prior=use_prior,
    )
    return r_new, {"iters": n_used}
