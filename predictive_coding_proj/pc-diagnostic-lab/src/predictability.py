"""C9 — held-out predictability (R^2 vs copy-last) and translation consistency.

Distinguishes "the code moved a lot" (cosine/L2 distance, see C5) from "the
code is not a learnable function of its past". Two reusable measurements:

  - predictability_r2: fit a tiny per-layer predictor from a context window of
    codes to the next residual, evaluate against a held-out copy-last floor.
  - translation_consistency: for a pure image translation, check whether the
    code approximately co-translates (rolls) at its own stride, i.e. whether
    the dictionary's equivariance survives settle.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference import settle_grounded
from src.metrics import pair_stats


def make_settle_encoder(deconvs, r_init, cfg, warm=True, init_noise=None, iters=None):
    """Stateful settle_grounded wrapper: encode_fn(frame) -> list[Tensor] (per-layer code).

    When warm=True the previous frame's settled code is used as r_warm for the
    next call (matching the pipeline protocol: fixed r_init on the first
    frame, warm-start afterwards). Call .reset() before a new sequence.
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    alpha = inf["alpha"]
    lr_r = inf["lr_r"]
    sigma_2 = inf["sigma_2"]
    num_epochs_inner = inf["num_epochs_inner"] if iters is None else iters
    use_prior = inf.get("use_prior", True)
    conv_tol = inf.get("conv_tol", None)
    noise = inf.get("init_noise", 0.01) if init_noise is None else init_noise

    state = {"r_prev": None}

    def encode(frame):
        r_warm = state["r_prev"] if (warm and state["r_prev"] is not None) else None
        r, _ = settle_grounded(
            frame,
            r_init,
            deconvs,
            alpha,
            lr_r,
            sigma_2,
            num_epochs_inner,
            num_layers,
            r_warm=r_warm,
            conv_tol=conv_tol,
            init_noise=noise,
            use_prior=use_prior,
        )
        if warm:
            state["r_prev"] = r
        return r

    def reset():
        state["r_prev"] = None

    encode.reset = reset
    return encode


def encode_sequences(seqs, encode_fn):
    """seqs: list of [T,1,H,W] mean-centred sequences.

    Calls encode_fn.reset() (if present) before each sequence, then
    encode_fn(frame[1,1,H,W]) in order t=0..T-1.

    Returns list[list[list[Tensor]]]: per sequence, per frame, per-layer code.
    """
    out = []
    for seq in seqs:
        if hasattr(encode_fn, "reset"):
            encode_fn.reset()
        codes = []
        T = seq.shape[0]
        for t in range(T):
            frame = seq[t]
            if frame.ndim == 3:
                frame = frame.unsqueeze(0)
            codes.append(encode_fn(frame))
        out.append(codes)
    return out


def pixel_codes(seqs):
    """Wrap each frame as a one-layer "code" so predictability_r2 runs on pixels."""
    out = []
    for seq in seqs:
        codes = []
        T = seq.shape[0]
        for t in range(T):
            frame = seq[t]
            if frame.ndim == 3:
                frame = frame.unsqueeze(0)
            codes.append([frame])
        out.append(codes)
    return out


def _build_windows(codes, context):
    """codes: list[list[list[Tensor]]] -> per-layer stacked (input, target) windows.

    input window at t: concat(codes[t-context:t], layer l) along channel dim.
    target at t: codes[t][l] - codes[t-1][l] (residual).
    """
    num_layers = len(codes[0][0])
    inputs = [[] for _ in range(num_layers)]
    targets = [[] for _ in range(num_layers)]
    for seq in codes:
        T = len(seq)
        for t in range(context, T):
            for l in range(num_layers):
                ctx = torch.cat([seq[t - context + k][l] for k in range(context)], dim=1)
                tgt = seq[t][l] - seq[t - 1][l]
                inputs[l].append(ctx)
                targets[l].append(tgt)
    stacked_in = [torch.cat(x, dim=0) for x in inputs]
    stacked_tgt = [torch.cat(x, dim=0) for x in targets]
    return stacked_in, stacked_tgt, num_layers


def _build_model(in_ch, out_ch, hidden, model):
    if model == "linear":
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    if model == "conv":
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_ch, kernel_size=3, padding=1),
        )
    raise ValueError(f"unknown predictability model '{model}' (expected 'linear' or 'conv')")


def predictability_r2(train_codes, test_codes, device, context=2, steps=300, lr=1e-3, hidden=32, seed=0, model="conv"):
    """Fit a small per-layer predictor (r_{t-context}..r_{t-1}) -> (r_t - r_{t-1}).

    Full-batch Adam for `steps` steps on all train windows at once, per layer
    independently, evaluated on held-out test windows against the copy-last
    (residual = 0) and mean-train-residual baselines.
    """
    train_in, train_tgt, num_layers = _build_windows(train_codes, context)
    test_in, test_tgt, _ = _build_windows(test_codes, context)

    per_layer = []
    n_train_windows = None
    n_test_windows = None
    for l in range(num_layers):
        x_tr, y_tr = train_in[l].to(device), train_tgt[l].to(device)
        x_te, y_te = test_in[l].to(device), test_tgt[l].to(device)
        n_train_windows = int(x_tr.shape[0])
        n_test_windows = int(x_te.shape[0])
        in_ch, out_ch = x_tr.shape[1], y_tr.shape[1]

        torch.manual_seed(seed * 1000 + l)
        net = _build_model(in_ch, out_ch, hidden, model).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(net(x_tr), y_tr)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred_test = net(x_te)
            mse_pred = float(F.mse_loss(pred_test, y_te).item())
            mse_copy_last = float(torch.mean(y_te ** 2).item())
            mean_train_residual = y_tr.mean(dim=0, keepdim=True)
            mse_mean_target = float(F.mse_loss(mean_train_residual.expand_as(y_te), y_te).item())

        r2_copy = float(1.0 - mse_pred / (mse_copy_last + 1e-12))
        r2_mean = float(1.0 - mse_pred / (mse_mean_target + 1e-12))
        numel = int(y_te.numel())
        per_layer.append(
            {
                "layer": l,
                "mse_pred": mse_pred,
                "mse_copy_last": mse_copy_last,
                "r2_vs_copy_last": r2_copy,
                "mse_mean_target": mse_mean_target,
                "r2_vs_mean": r2_mean,
                "numel": numel,
            }
        )

    total_numel = sum(p["numel"] for p in per_layer)

    def wmean(key):
        return float(sum(p[key] * p["numel"] for p in per_layer) / total_numel)

    mse_pred = wmean("mse_pred")
    mse_copy_last = wmean("mse_copy_last")
    mse_mean_target = wmean("mse_mean_target")
    return {
        "mse_pred": mse_pred,
        "mse_copy_last": mse_copy_last,
        "r2_vs_copy_last": float(1.0 - mse_pred / (mse_copy_last + 1e-12)),
        "mse_mean_target": mse_mean_target,
        "r2_vs_mean": float(1.0 - mse_pred / (mse_mean_target + 1e-12)),
        "per_layer": per_layer,
        "n_train_windows": n_train_windows,
        "n_test_windows": n_test_windows,
        "context": context,
        "model": model,
    }


def translation_consistency(frames, encode_fn, shifts, strides):
    """For each (dx, dy) shift, check whether the code co-translates with the image.

    frames: list of [1,1,H,W] mean-centred frames.
    encode_fn: cold, deterministic encoder (e.g. make_settle_encoder(..., warm=False,
      init_noise=0.0)); reset() is called before every settle if available.
    strides[l]: pixels-per-code-cell at layer l.

    Returns {"per_shift": [{"dx","dy","pixel_cos","layers":[{"layer","cos","rel",
      "exact_expected","aliased"}...]}...]}.
    """
    num_layers = len(strides)
    per_shift = []
    for dx, dy in shifts:
        layer_cos = [[] for _ in range(num_layers)]
        layer_rel = [[] for _ in range(num_layers)]
        layer_exact = [None] * num_layers
        pixel_cos_list = []
        for frame in frames:
            shifted = torch.roll(frame, shifts=(dy, dx), dims=(2, 3))

            if hasattr(encode_fn, "reset"):
                encode_fn.reset()
            code_a = encode_fn(frame)
            if hasattr(encode_fn, "reset"):
                encode_fn.reset()
            code_b = encode_fn(shifted)

            pixel_cos_list.append(pair_stats(shifted, frame)["cos"])
            for l in range(num_layers):
                s = strides[l]
                exact = (dx % s == 0) and (dy % s == 0)
                if exact:
                    sy, sx = dy // s, dx // s
                else:
                    sy, sx = round(dy / s), round(dx / s)
                rolled_a = torch.roll(code_a[l], shifts=(sy, sx), dims=(2, 3))
                stats = pair_stats(code_b[l], rolled_a)
                layer_cos[l].append(stats["cos"])
                layer_rel[l].append(stats["rel"])
                layer_exact[l] = exact

        per_shift.append(
            {
                "dx": dx,
                "dy": dy,
                "pixel_cos": float(np.mean(pixel_cos_list)),
                "layers": [
                    {
                        "layer": l,
                        "cos": float(np.mean(layer_cos[l])),
                        "rel": float(np.mean(layer_rel[l])),
                        "exact_expected": bool(layer_exact[l]),
                        "aliased": not bool(layer_exact[l]),
                    }
                    for l in range(num_layers)
                ],
            }
        )
    return {"per_shift": per_shift}
