"""Per-term PC energy + the C12 rollout/control machinery that consumes it.

Owned by Agent B. `energy_terms` is the shared primitive (spec §"Per-term
energy"): it reuses the exact e0/e1/prior/total formulas from
`src.inference.settle_info` (that file is not modified here) and additionally
splits out the DC-offset artifact described in SPEC2's background section —
closed-loop decoded frames drift to a nonzero mean, and because
`_prepare_image` centers the TARGET but never the decode, that drift shows up
as pure "reconstruction error" with no shape information in it. `dc_offset`
reports the raw drift; `e0_dc_free` is the image term with the drift removed
from both sides, i.e. what's left once you stop conflating "wrong DC" with
"wrong shape".

`energy_fade_rollout` / `aggregate_energy_fade` are evolved copies of the
same-named functions in `src.offline_gru` (owned by Agent A) — copied here
per SPEC2 rather than edited in place, then reworked to log the honest
per-term signal (e1, e0_dc_free) instead of a single energy total that hides
the artifact. `contrast_sweep` is the new artificial-fade control: it shows
energy is monotone in raw contrast with no rollout involved at all, which is
the baseline every "does energy track fade" claim has to be read against.
"""

import torch

from src.inference import settle_grounded
from src.metrics import flat_r, pair_stats, stack_mean_std
from src.predictability import make_settle_encoder
from src.spatial_pc import f_clamp, make_variables
from src.utils import clone_r

# ---------------------------------------------------------------------------
# Per-term energy
# ---------------------------------------------------------------------------


def energy_terms(I, r, deconvs, alpha, sigma_2, num_layers, use_prior=True, recenter=True):
    """PC energy at r, decomposed per term, plus the DC-offset artifact.

    e0, e1, prior, total: identical formulas/values to `settle_info` in
    src/inference.py (e0 = layer-0 image term, e1 = sum of the inter-layer
    terms for layers 1..num_layers-1 — a single "e1" bucket since this project
    only ever runs num_layers=2, so there is exactly one inter-layer term).

    dc_offset: mean of the RAW decoded frame f_clamp(deconv0(r0)), before any
    centering — the quantity that leaks into e0 as an artifact when the
    target I is centered but the decode has drifted off zero mean.

    e0_dc_free: the image term with BOTH I and the decode re-centered by
    their own means before differencing. Any constant DC mismatch between
    target and decode cancels out here, so this is the reconstruction error
    that is actually about shape. (e0 - e0_dc_free) is the DC artifact's
    contribution to e0.

    All under no_grad; does not mutate r.
    """
    with torch.no_grad():
        I_curr = I.float()
        if I_curr.ndim == 3:
            I_curr = I_curr.unsqueeze(0)
        r = [ri.detach().clone() for ri in r]

        D0 = f_clamp(deconvs[0](r[0]))
        dc_offset = float(D0.mean().item())

        I_target = (I_curr - I_curr.mean()) if recenter else I_curr
        e_spatial = [None] * num_layers
        e_spatial[0] = I_target - D0
        for i in range(1, num_layers):
            e_spatial[i] = r[i - 1] - f_clamp(deconvs[i](r[i]))

        e0 = float(torch.sum(e_spatial[0] ** 2).item() / (2 * sigma_2))
        if num_layers > 1:
            e1 = float(
                sum(torch.sum(e_spatial[i] ** 2).item() for i in range(1, num_layers)) / (2 * sigma_2)
            )
        else:
            e1 = 0.0

        I_dc = I_curr - I_curr.mean()
        D0_dc = D0 - D0.mean()
        e0_dc_free = float(torch.sum((I_dc - D0_dc) ** 2).item() / (2 * sigma_2))

        prior = 0.0
        if use_prior:
            for i in range(num_layers):
                prior += alpha * float(torch.sum(torch.log(1 + r[i] ** 2)).item())
        prior = float(prior)

        total = float(e0 + e1 + prior)

    return {
        "e0": e0,
        "e1": e1,
        "prior": prior,
        "total": total,
        "dc_offset": dc_offset,
        "e0_dc_free": e0_dc_free,
    }


def _check_energy_terms(device="cpu"):
    """Self-check: total == e0+e1+prior, and the DC artifact matches the
    closed-form N*mean^2/2 while e0_dc_free is ~0 for a decoded-frame target.

    Raises AssertionError on failure. Returns a short report string on success.
    """
    torch.manual_seed(0)
    num_layers = 2
    size = 64  # 64*64 = 4096 pixels, matching the background's "4096*mean^2/2".
    image = torch.zeros(1, 1, size, size, device=device)
    _, r_init, deconvs = make_variables(image, initial_r_size=size, num_layers=num_layers, device=device)
    r = [torch.randn_like(ri) * 0.3 for ri in r_init]
    alpha, sigma_2 = 0.001, 1.0

    # 1) total == e0 + e1 + prior on an arbitrary random code / random frame.
    I_rand = torch.randn(1, 1, size, size, device=device) * 0.2
    terms = energy_terms(I_rand, r, deconvs, alpha, sigma_2, num_layers, use_prior=True)
    lhs, rhs = terms["total"], terms["e0"] + terms["e1"] + terms["prior"]
    assert abs(lhs - rhs) < 1e-4 * max(1.0, abs(rhs)), f"total {lhs} != e0+e1+prior {rhs}"

    # 2) DC artifact: I built as the decoded frame's own mean subtracted off,
    #    so e0 (which centers only I, not the decode) sees exactly the DC
    #    offset as error, while e0_dc_free (centers both) sees none.
    with torch.no_grad():
        D0 = f_clamp(deconvs[0](r[0]))
    I_dc = D0 - D0.mean()
    dc_terms = energy_terms(I_dc, r, deconvs, alpha, sigma_2, num_layers, use_prior=True)
    n_pixels = I_dc.numel()
    expected_e0 = n_pixels * (dc_terms["dc_offset"] ** 2) / (2 * sigma_2)
    assert abs(dc_terms["e0"] - expected_e0) < 1e-3 * max(1.0, expected_e0), (
        f"e0 {dc_terms['e0']} != N*mean^2/2 {expected_e0}"
    )
    assert dc_terms["e0_dc_free"] < 1e-6, f"e0_dc_free {dc_terms['e0_dc_free']} not ~0"

    report = (
        f"_check_energy_terms OK: total==e0+e1+prior ({lhs:.6f}); "
        f"DC artifact e0={dc_terms['e0']:.4f} ~= N*mean^2/2={expected_e0:.4f} "
        f"(dc_offset={dc_terms['dc_offset']:.4f}), e0_dc_free={dc_terms['e0_dc_free']:.2e}"
    )
    print(report)
    return report


# ---------------------------------------------------------------------------
# Artificial-fade control (no rollout involved)
# ---------------------------------------------------------------------------


def contrast_sweep(frames, r_init, deconvs, cfg, scales):
    """Settle from zero init directly on scale*I for each frame/scale.

    This is deliberately rollout-free: it isolates whether the energy
    function itself is monotone in raw pixel contrast, independent of any
    GRU rollout. `blank_is_minimum` in the aggregation answers "is scale=0
    (a blank frame) the global energy minimum" — SPEC2's background finding
    that it is, which means "energy doesn't rise as it fades" can be a
    property of the energy function rather than of the rollout.

    Returns {scale: {term_or_"r_norm": [values, one per frame]}}.
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    alpha, sigma_2 = inf["alpha"], inf["sigma_2"]
    use_prior = inf.get("use_prior", True)
    c12 = cfg.get("c12", {})
    settle_iters = c12.get("settle_iters", 200)
    lr_r = inf["lr_r"]

    term_keys = ("e0", "e1", "prior", "total", "dc_offset", "e0_dc_free")
    per_scale = {s: {k: [] for k in term_keys + ("r_norm",)} for s in scales}

    for I in frames:
        I = I.float()
        if I.ndim == 3:
            I = I.unsqueeze(0)
        for s in scales:
            I_s = s * I
            r_s, _ = settle_grounded(
                I_s,
                r_init,
                deconvs,
                alpha,
                lr_r,
                sigma_2,
                settle_iters,
                num_layers,
                init_noise=0.0,
                use_prior=use_prior,
            )
            terms = energy_terms(I_s, r_s, deconvs, alpha, sigma_2, num_layers, use_prior=use_prior)
            for k in term_keys:
                per_scale[s][k].append(terms[k])
            per_scale[s]["r_norm"].append(float(flat_r(r_s).norm().item()))

    return per_scale


def aggregate_contrast_sweep(per_scale, scales):
    """Mean/std per scale per term, plus the blank_is_minimum verdict.

    per_scale may be pre-pooled across seeds (caller extends the raw lists
    before aggregating once) or a single seed's output.
    """
    term_keys = ("e0", "e1", "prior", "total", "dc_offset", "e0_dc_free", "r_norm")
    scales_asc = sorted(scales)
    per_scale_out = {}
    for s in scales_asc:
        d = per_scale[s]
        entry = {}
        for k in term_keys:
            xs = d.get(k, [])
            arr = torch.tensor(xs, dtype=torch.float64) if xs else torch.tensor([float("nan")])
            entry[f"{k}_mean"] = float(arr.mean().item()) if xs else float("nan")
            entry[f"{k}_std"] = float(arr.std(unbiased=False).item()) if len(xs) > 1 else 0.0
        entry["n"] = len(d.get("total", []))
        per_scale_out[str(s)] = entry

    totals_asc = [per_scale_out[str(s)]["total_mean"] for s in scales_asc]
    blank_is_minimum = bool(
        len(totals_asc) > 1 and all(totals_asc[i] < totals_asc[i + 1] for i in range(len(totals_asc) - 1))
    )

    return {
        "per_scale": per_scale_out,
        "scales": scales_asc,
        "blank_is_minimum": blank_is_minimum,
        "energy_at_scale": {str(s): per_scale_out[str(s)]["total_mean"] for s in scales_asc},
    }


# ---------------------------------------------------------------------------
# Rollout logging
# ---------------------------------------------------------------------------


def _prepare_frame(frame):
    x = frame.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x - x.mean()


def _contrast(frame):
    x = frame.detach()
    return float(x.abs().mean().item()), float(x.abs().amax().item())


def energy_fade_rollout(seq, r_init, deconvs, temporal_nn, cfg, split_point):
    """Pure-latent closed-loop after `split_point`; per-term energy of true
    vs predicted-vs-own-decode vs re-settled-on-decode, at every t>=1.

    Before the handoff the GRU is teacher-forced on eval-protocol codes.
    After it, r_prev <- r_pred (no re-settle). At every t>=1 we log:

      true    energy_terms(I_t, r_true_t)               -- the true frame/code.
      pred    energy_terms(I_hat_t, r_pred_t)            -- predicted code vs
                                                             its OWN decode (no
                                                             extra settle): this
                                                             is where the DC
                                                             artifact lives, now
                                                             visible as
                                                             pred.dc_offset /
                                                             (pred.e0 - pred.e0_dc_free)
                                                             instead of hidden
                                                             in a single total.
      faded   energy_terms(I_hat_t, r_faded_t)           -- predicted code
                                                             re-settled on its
                                                             own decode (information
                                                             -free: the decode is
                                                             a function of r_pred,
                                                             so this settle has
                                                             nothing new to find).

    plus mass/peak of true vs decoded frames, cos(r_pred, r_true), norms,
    e1_true (convenience alias for true["e1"]), and the contrast-normalised
    e_norm = total / (||I||^2 + eps) for both true and predicted frames (so
    a rise in energy can be told apart from a rise in raw contrast).
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    alpha, sigma_2 = inf["alpha"], inf["sigma_2"]
    use_prior = inf.get("use_prior", True)
    term_kw = dict(deconvs=deconvs, alpha=alpha, sigma_2=sigma_2, num_layers=num_layers, use_prior=use_prior)
    settle_kw = dict(
        r_init=r_init,
        deconvs=deconvs,
        alpha=alpha,
        lr_r=inf["lr_r"],
        sigma_2=sigma_2,
        num_epochs_inner=inf["num_epochs_inner"],
        num_layers=num_layers,
        init_noise=inf.get("init_noise", 0.01),
        use_prior=use_prior,
    )

    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]

    encoder = make_settle_encoder(deconvs, r_init, cfg, warm=True)
    encoder.reset()
    r_true, I_true = [], []
    for t in range(T):
        I = _prepare_frame(seq[t])
        I_true.append(I)
        r_true.append(encoder(I))

    hidden = temporal_nn.init_hidden(r_init)
    r_prev = None
    steps = []
    temporal_nn.eval()
    eps = 1e-8
    with torch.no_grad():
        for t in range(T):
            r_in = [torch.zeros_like(ri) for ri in r_true[0]] if r_prev is None else r_prev
            r_pred, hidden, _ = temporal_nn(r_in, hidden)
            if t >= 1:
                I_hat = f_clamp(deconvs[0](r_pred[0]))
                I_hat = I_hat - I_hat.mean()
                r_faded, _ = settle_grounded(I_hat, r_warm=r_pred, **settle_kw)

                true_terms = energy_terms(I_true[t], r_true[t], **term_kw)
                pred_terms = energy_terms(I_hat, r_pred, **term_kw)
                faded_terms = energy_terms(I_hat, r_faded, **term_kw)

                mass_t, peak_t = _contrast(I_true[t])
                mass_p, peak_p = _contrast(I_hat)

                norm_true_I = float(torch.sum(I_true[t] ** 2).item())
                norm_pred_I = float(torch.sum(I_hat ** 2).item())

                steps.append(
                    {
                        "t": t,
                        "closed_loop": t >= split_point,
                        "true": true_terms,
                        "pred": pred_terms,
                        "faded": faded_terms,
                        "energy_pred_code": pred_terms["total"],  # backward-compat alias
                        "e1_true": true_terms["e1"],
                        "e_norm_true": true_terms["total"] / (norm_true_I + eps),
                        "e_norm_pred": pred_terms["total"] / (norm_pred_I + eps),
                        "mass_true": mass_t,
                        "mass_pred": mass_p,
                        "peak_true": peak_t,
                        "peak_pred": peak_p,
                        "cos_pred_true": pair_stats(r_pred, r_true[t])["cos"],
                        "r_pred_norm": float(flat_r(r_pred).norm().item()),
                        "r_true_norm": float(flat_r(r_true[t]).norm().item()),
                    }
                )
            hidden = [hi.detach() for hi in hidden]
            r_prev = clone_r(r_pred) if t >= split_point else clone_r(r_true[t])

    return {"steps": steps, "split_point": split_point}


_TERM_SUBKEYS = ("e0", "e1", "prior", "total", "dc_offset", "e0_dc_free")
_FLAT_SCALAR_KEYS = (
    "energy_pred_code",
    "e1_true",
    "e_norm_true",
    "e_norm_pred",
    "mass_true",
    "mass_pred",
    "peak_true",
    "peak_pred",
    "cos_pred_true",
    "r_pred_norm",
    "r_true_norm",
)


def _flatten_step(s):
    flat = {"t": s["t"], "closed_loop": s["closed_loop"]}
    for prefix in ("true", "pred", "faded"):
        d = s[prefix]
        for k in _TERM_SUBKEYS:
            flat[f"{prefix}_{k}"] = d[k]
    for k in _FLAT_SCALAR_KEYS:
        flat[k] = s[k]
    return flat


def _flat_keys():
    keys = []
    for prefix in ("true", "pred", "faded"):
        keys.extend(f"{prefix}_{k}" for k in _TERM_SUBKEYS)
    keys.extend(_FLAT_SCALAR_KEYS)
    return keys


def aggregate_energy_fade(per_seq, split_point):
    """Mean/std curves across sequences plus pre/post-split summaries and verdicts.

    per_seq: list of {"steps": [...], ...} as returned by energy_fade_rollout
    (may be pooled across seeds by the caller — stack_mean_std just needs
    equal-length step lists, which holds as long as every sequence has the
    same length and split_point).
    """
    if not per_seq:
        return {"curves": {}, "pre_split": {}, "post_split": {}}

    flat_keys = _flat_keys()
    flat_seq = [[_flatten_step(s) for s in seq["steps"]] for seq in per_seq]

    curves = {}
    for k in flat_keys:
        mu, sd = stack_mean_std([[fs[k] for fs in seq] for seq in flat_seq])
        curves[f"{k}_mean"] = mu
        curves[f"{k}_std"] = sd

    def _side(pred_fn):
        vals = {k: [] for k in flat_keys}
        for seq in flat_seq:
            for fs in seq:
                if pred_fn(fs):
                    for k in flat_keys:
                        vals[k].append(fs[k])
        out = {}
        for k, xs in vals.items():
            arr = torch.tensor(xs, dtype=torch.float64) if xs else torch.tensor([float("nan")])
            out[k] = {
                "mean": float(arr.mean().item()) if xs else float("nan"),
                "std": float(arr.std(unbiased=False).item()) if len(xs) > 1 else 0.0,
                "n": len(xs),
            }
        return out

    pre = _side(lambda fs: not fs["closed_loop"])
    post = _side(lambda fs: fs["closed_loop"])

    eps = 1e-8

    def _e1_ratio(side):
        e1_pred = side["pred_e1"]["mean"]
        e1_true = side["true_e1"]["mean"]
        return float(e1_pred / (e1_true + eps))

    e1_ratio_post = _e1_ratio(post)
    e1_ratio_pre = _e1_ratio(pre)

    e0_p = post["pred_e0"]["mean"]
    e0_dc_p = post["pred_e0_dc_free"]["mean"]
    dc_artifact_share_post = float((e0_p - e0_dc_p) / (e0_p + eps))

    # fade_then_diverge: on the pooled mean curve, does the predicted norm dip
    # below 0.8x its pre-split mean at some post-split step, and later exceed
    # the true norm?
    pre_norm_mean = pre["r_pred_norm"]["mean"]
    t_list = [fs["t"] for fs in flat_seq[0]] if flat_seq and flat_seq[0] else []
    r_pred_curve = curves.get("r_pred_norm_mean") or []
    r_true_curve = curves.get("r_true_norm_mean") or []
    fade_then_diverge = False
    dipped = False
    for i, t in enumerate(t_list):
        if t < split_point or i >= len(r_pred_curve) or i >= len(r_true_curve):
            continue
        if not dipped and r_pred_curve[i] < 0.8 * pre_norm_mean:
            dipped = True
        elif dipped and r_pred_curve[i] > r_true_curve[i]:
            fade_then_diverge = True
            break

    e_norm_pred_pre = pre["e_norm_pred"]["mean"]
    e_norm_pred_post = post["e_norm_pred"]["mean"]
    energy_norm_tracks_fade = bool(e_norm_pred_post > e_norm_pred_pre)

    return {
        "curves": curves,
        "pre_split": pre,
        "post_split": post,
        "split_point": split_point,
        "n_sequences": len(per_seq),
        "e1_ratio_post": e1_ratio_post,
        "e1_ratio_pre": e1_ratio_pre,
        "dc_artifact_share_post": dc_artifact_share_post,
        "fade_then_diverge": bool(fade_then_diverge),
        "energy_norm_tracks_fade": energy_norm_tracks_fade,
        "e_norm_pred_pre": e_norm_pred_pre,
        "e_norm_pred_post": e_norm_pred_post,
    }


if __name__ == "__main__":
    _check_energy_terms()
