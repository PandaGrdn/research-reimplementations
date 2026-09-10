"""Teacher-forced and long-horizon hierarchical validation (notebook spelling kept as alias)."""

import torch
import torch.nn.functional as F

from src.metrics import delta_ratio, long_horizon_mean, motion_gap, saturation_frac
from src.spatial_pc import f_clamp
from src.utils import clone_r


def _prepare_seq(seq):
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    return seq


def validate_hierarchical(
    seq,
    r_init,
    num_epochs_inner,
    num_layers,
    sigma_2,
    alpha,
    lr_r,
    deconvs,
    temporal_nn,
    use_prior=True,
    temporal_prior_weight=0.01,
    encoder=None,
    log=print,
):
    """Teacher-forced next-frame validation (notebook validate_heirarchical)."""
    seq = _prepare_seq(seq)
    T = seq.shape[0]
    e_spatial = [None] * num_layers

    r_prev1 = None
    hidden = temporal_nn.init_hidden(r_init)
    r_curr = clone_r(r_init)

    true_frames = []
    pred_frames = []
    all_curr = []
    I_prev = None
    delta_norms, dr_norms = [], []
    mse_vs_curr, mse_vs_prev = [], []

    for t in range(T):
        I_curr = seq[t] - seq[t].mean()
        if I_curr.ndim == 3:
            I_curr = I_curr.unsqueeze(0)

        if r_prev1 is None:
            r_prev1_use = [torch.zeros_like(ri) for ri in r_curr]
        else:
            r_prev1_use = [ri.detach() for ri in r_prev1]

        with torch.no_grad():
            r_pred, h_new, deltas = temporal_nn(r_prev1_use, hidden)
            if t >= 1:
                I_hat = f_clamp(deconvs[0](r_pred[0]))
                true_frames.append(I_curr.detach().clone())
                pred_frames.append(I_hat.detach().clone())
                mse_vs_curr.append(torch.mean((I_hat - I_curr) ** 2).item())
                if I_prev is not None:
                    mse_vs_prev.append(torch.mean((I_hat - I_prev) ** 2).item())

            if encoder is not None:
                r_curr = [ri.detach() for ri in encoder(I_curr)]
            else:
                for _ in range(num_epochs_inner):
                    e_spatial[0] = I_curr - f_clamp(deconvs[0](r_curr[0]))
                    for i in range(1, num_layers):
                        e_spatial[i] = r_curr[i - 1] - f_clamp(deconvs[i](r_curr[i]))
                    for i in range(num_layers):
                        cauchy_prior = alpha * (2 * r_curr[i] / (1 + r_curr[i] ** 2)) if use_prior else 0.0
                        bottom_up = F.conv2d(
                            e_spatial[i], deconvs[i].weight, padding=deconvs[i].padding, stride=deconvs[i].stride
                        )
                        dr = (1.0 / sigma_2) * bottom_up - cauchy_prior - (temporal_prior_weight / sigma_2) * (
                            r_curr[i] - r_pred[i]
                        )
                        if i < num_layers - 1:
                            dr = dr - (1.0 / sigma_2) * e_spatial[i + 1]
                        r_curr[i] = (r_curr[i] + lr_r * dr).detach()

            if t >= 1:
                d_norm = 0.0
                a_norm = 0.0
                for i in range(num_layers):
                    d_norm += torch.norm(deltas[i]).item()
                    a_norm += torch.norm(r_curr[i] - r_prev1_use[i]).item()
                delta_norms.append(d_norm)
                dr_norms.append(a_norm)

        hidden = [hi.detach() for hi in h_new]
        r_prev1 = clone_r(r_curr)
        all_curr.append(I_curr.detach().clone())
        I_prev = I_curr.detach().clone()

    true_frames = torch.stack(true_frames, dim=0)
    pred_frames = torch.stack(pred_frames, dim=0)
    mse = torch.mean((true_frames - pred_frames) ** 2)
    mse_per_frame = torch.mean((true_frames - pred_frames) ** 2, dim=(1, 2, 3, 4))
    ratio, mean_delta, mean_dr = delta_ratio(delta_norms, dr_norms)
    gap, mean_mse_curr, mean_mse_prev = motion_gap(mse_vs_curr, mse_vs_prev)

    copy_last_mse_per_frame, mean_frame_mse_per_frame = [], []
    frame_sum = None
    for t in range(1, T):
        frame_sum = all_curr[t - 1] if frame_sum is None else frame_sum + all_curr[t - 1]
        mean_ref = frame_sum / t
        copy_last_mse_per_frame.append(float(torch.mean((all_curr[t] - all_curr[t - 1]) ** 2).item()))
        mean_frame_mse_per_frame.append(float(torch.mean((all_curr[t] - mean_ref) ** 2).item()))
    copy_last_mse = float(sum(copy_last_mse_per_frame) / max(len(copy_last_mse_per_frame), 1))
    mean_frame_mse = float(sum(mean_frame_mse_per_frame) / max(len(mean_frame_mse_per_frame), 1))

    log(
        f"Identity check | ||delta||={mean_delta:.6f}  ||r_t-r_(t-1)||={mean_dr:.6f}  "
        f"ratio={ratio:.4f}  delta_scale={temporal_nn.delta_scale}"
    )
    if mean_mse_curr < 0.1:
        log(
            f"Copy-last check | MSE(pred,true_t)={mean_mse_curr:.6f}  "
            f"MSE(pred,true_(t-1))={mean_mse_prev:.6f}  gap={gap:+.6f}  "
            f"{'COPY' if gap > 0 else 'MOTION'}"
        )
    else:
        log(f"Copy-last check | MSE(pred,true_t)={mean_mse_curr:.6f}  (skip COPY/MOTION — preds not digit-like yet)")

    return {
        "mse": float(mse.detach()),
        "mse_per_frame": [float(x) for x in mse_per_frame.detach().cpu()],
        "true_frames": true_frames,
        "pred_frames": pred_frames,
        "r_curr": r_curr,
        "motion_gap": gap,
        "delta_ratio": ratio,
        "mean_mse_curr": mean_mse_curr,
        "mean_mse_prev": mean_mse_prev,
        "copy_last": gap > 0,
        "copy_last_mse_per_frame": copy_last_mse_per_frame,
        "mean_frame_mse_per_frame": mean_frame_mse_per_frame,
        "copy_last_mse": copy_last_mse,
        "mean_frame_mse": mean_frame_mse,
    }


def validate_hierarchical_long(
    seq,
    r_init,
    num_epochs_inner,
    num_layers,
    sigma_2,
    alpha,
    lr_r,
    deconvs,
    temporal_nn,
    split_point=10,
    split_fix=False,
    use_prior=True,
    temporal_prior_weight=0.01,
    encoder=None,
    log=print,
):
    """Closed-loop rollout. split_fix selects the r_prev1 bookkeeping branch.

    original (split_fix=False): r_prev1 ← r_pred after the temporal step
    fixed    (split_fix=True):  r_prev1 ← r_curr settled from I_obs (notebook cell 18)
    """
    seq = _prepare_seq(seq)
    T = seq.shape[0]
    e_spatial = [None] * num_layers

    r_prev1 = None
    hidden = temporal_nn.init_hidden(r_init)
    r_curr = clone_r(r_init)
    true_frames, pred_frames = [], []
    all_curr = []
    saturation = None

    for t in range(T):
        I_curr = seq[t] - seq[t].mean()
        if I_curr.ndim == 3:
            I_curr = I_curr.unsqueeze(0)

        if r_prev1 is None:
            r_prev1_use = [torch.zeros_like(ri) for ri in r_curr]
        else:
            r_prev1_use = [ri.detach() for ri in r_prev1]

        with torch.no_grad():
            r_pred, h_new, deltas = temporal_nn(r_prev1_use, hidden)
            if t >= 1:
                I_hat = f_clamp(deconvs[0](r_pred[0]))
                true_frames.append(I_curr.detach().clone())
                pred_frames.append(I_hat.detach().clone())

            if t >= split_point:
                I_obs = pred_frames[t - 1] - pred_frames[t - 1].mean()
                if I_obs.ndim == 3:
                    I_obs = I_obs.unsqueeze(0)
                if t == split_point:
                    saturation = {
                        "t": t,
                        "true": saturation_frac(I_curr),
                        "substitute": saturation_frac(I_obs),
                    }
                    log(f"t={t} saturation | true: {saturation['true']:.4f} | substitute: {saturation['substitute']:.4f}")
            else:
                I_obs = I_curr

            if split_fix or t < split_point:
                if encoder is not None:
                    r_curr = [ri.detach() for ri in encoder(I_obs)]
                else:
                    for _ in range(num_epochs_inner):
                        e_spatial[0] = I_obs - f_clamp(deconvs[0](r_curr[0]))
                        for i in range(1, num_layers):
                            e_spatial[i] = r_curr[i - 1] - f_clamp(deconvs[i](r_curr[i]))
                        for i in range(num_layers):
                            cauchy_prior = alpha * (2 * r_curr[i] / (1 + r_curr[i] ** 2)) if use_prior else 0.0
                            bottom_up = F.conv2d(
                                e_spatial[i], deconvs[i].weight, padding=deconvs[i].padding, stride=deconvs[i].stride
                            )
                            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior - (temporal_prior_weight / sigma_2) * (
                                r_curr[i] - r_pred[i]
                            )
                            if i < num_layers - 1:
                                dr = dr - (1.0 / sigma_2) * e_spatial[i + 1]
                            r_curr[i] = (r_curr[i] + lr_r * dr).detach()
            else:
                r_curr = clone_r(r_pred)

        hidden = [hi.detach() for hi in h_new]
        if (not split_fix) and t >= split_point:
            r_prev1 = clone_r(r_pred)
        else:
            r_prev1 = clone_r(r_curr)
        all_curr.append(I_curr.detach().clone())

    true_frames = torch.stack(true_frames, dim=0)
    pred_frames = torch.stack(pred_frames, dim=0)
    mse = torch.mean((true_frames - pred_frames) ** 2)
    mse_per_frame = torch.mean((true_frames - pred_frames) ** 2, dim=(1, 2, 3, 4))
    curve = [float(x) for x in mse_per_frame.detach().cpu()]

    mean_ref_long = torch.stack(all_curr[:split_point], dim=0).mean(dim=0)
    copy_last_mse_per_frame, mean_frame_mse_per_frame = [], []
    for t in range(1, T):
        ref_idx = min(t - 1, split_point - 1)
        copy_last_mse_per_frame.append(float(torch.mean((all_curr[t] - all_curr[ref_idx]) ** 2).item()))
        mean_frame_mse_per_frame.append(float(torch.mean((all_curr[t] - mean_ref_long) ** 2).item()))
    copy_last_mse = float(sum(copy_last_mse_per_frame) / max(len(copy_last_mse_per_frame), 1))
    mean_frame_mse = float(sum(mean_frame_mse_per_frame) / max(len(mean_frame_mse_per_frame), 1))
    copy_last_long_mse = long_horizon_mean(copy_last_mse_per_frame, split_point)
    mean_frame_long_mse = long_horizon_mean(mean_frame_mse_per_frame, split_point)

    return {
        "mse": float(mse.detach()),
        "mse_per_frame": curve,
        "true_frames": true_frames,
        "pred_frames": pred_frames,
        "r_curr": r_curr,
        "copy_last_mse_per_frame": copy_last_mse_per_frame,
        "mean_frame_mse_per_frame": mean_frame_mse_per_frame,
        "copy_last_mse": copy_last_mse,
        "mean_frame_mse": mean_frame_mse,
        "copy_last_long_mse": copy_last_long_mse,
        "mean_frame_long_mse": mean_frame_long_mse,
        "split_point": split_point,
        "split_fix": split_fix,
        "saturation": saturation,
        "long_mse": long_horizon_mean(curve, split_point),
    }


validate_heirarchical = validate_hierarchical
validate_heirarchical_long = validate_hierarchical_long
