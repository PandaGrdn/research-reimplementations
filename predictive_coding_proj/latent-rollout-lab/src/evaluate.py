from collections import defaultdict

import torch
import torch.nn.functional as F

from src.metrics import (
    LPIPSMeter,
    copy_last_frames,
    headline_mean,
    latent_cos,
    latent_l2,
    mean_context_frames,
    per_step_mse,
    per_step_ssim,
)


def _to_list(t):
    if torch.is_tensor(t):
        return [float(x) for x in t.detach().cpu().flatten().tolist()]
    return [float(x) for x in t]


@torch.no_grad()
def evaluate(model, ae, loader, context_lengths=(2, 8), horizons=(1, 5, 10, 18), lpips=True, device=None):
    """Returns nested dict:
    - per-step MSE/SSIM/LPIPS curves (AR rollout), context x horizon
    - teacher-forced 1-step baselines
    - copy-last-frame and static-blur reference scores (sanity floors)
    - latent-space drift: ||r_pred_t|| and cos(r_pred_t, r_true_t) per step
    """
    model.eval()
    ae.eval()
    device = device or next(model.parameters()).device
    lpips_meter = LPIPSMeter(device, enabled=lpips)

    acc = {}
    counts = {}
    recon_num, recon_den = 0.0, 0

    for frames, latents in loader:
        frames = frames.to(device)
        latents = latents.to(device)
        b, t_len = frames.shape[:2]
        recon_num += F.mse_loss(ae.decode(latents), frames).item() * b
        recon_den += b

        r_tf = model.predict_latent(latents, teacher_force=True)
        i_tf = ae.decode(r_tf)

        for context in context_lengths:
            if t_len - context < 1:
                continue
            n_steps = t_len - context
            key = str(context)
            if key not in acc:
                acc[key] = defaultdict(lambda: torch.zeros(n_steps))
                counts[key] = 0

            r_ar = model.rollout(latents[:, :context], n_steps=n_steps)
            i_ar = ae.decode(r_ar)
            target_i = frames[:, context:]
            target_r = latents[:, context:]
            i_tf_win = i_tf[:, context - 2 : context - 2 + n_steps]
            r_tf_win = r_tf[:, context - 2 : context - 2 + n_steps]
            copy_i = copy_last_frames(frames, context, n_steps)
            mean_i = mean_context_frames(frames, context, n_steps)

            acc[key]["ar_mse"] += per_step_mse(i_ar, target_i).cpu() * b
            acc[key]["ar_ssim"] += per_step_ssim(i_ar, target_i) * b
            acc[key]["ar_lpips"] += lpips_meter.per_step(i_ar, target_i) * b
            acc[key]["ar_latent_mse"] += per_step_mse(r_ar, target_r).cpu() * b
            acc[key]["ar_latent_l2"] += latent_l2(r_ar, target_r).cpu() * b
            acc[key]["ar_latent_cos"] += latent_cos(r_ar, target_r).cpu() * b
            acc[key]["ar_latent_norm"] += r_ar.flatten(2).norm(dim=2).mean(dim=0).cpu() * b
            acc[key]["tf_mse"] += per_step_mse(i_tf_win, target_i).cpu() * b
            acc[key]["tf_latent_mse"] += per_step_mse(r_tf_win, target_r).cpu() * b
            acc[key]["copy_mse"] += per_step_mse(copy_i, target_i).cpu() * b
            acc[key]["mean_mse"] += per_step_mse(mean_i, target_i).cpu() * b
            counts[key] += b

    out = {
        "ae_recon_mse": recon_num / max(recon_den, 1),
        "ar": {},
        "tf": {},
        "floors": {"copy_last": {}, "mean_frame": {}},
        "headline": {},
        "horizons": {},
    }
    for context, bag in acc.items():
        n = max(counts[context], 1)
        curves = {k: _to_list(v / n) for k, v in bag.items()}
        ctx = int(context)
        out["ar"][context] = {
            "mse": curves["ar_mse"],
            "ssim": curves["ar_ssim"],
            "lpips": curves["ar_lpips"],
            "latent_mse": curves["ar_latent_mse"],
            "latent_l2": curves["ar_latent_l2"],
            "latent_cos": curves["ar_latent_cos"],
            "latent_norm": curves["ar_latent_norm"],
        }
        out["tf"][context] = {
            "mse": curves["tf_mse"],
            "latent_mse": curves["tf_latent_mse"],
            "first_step_mse": curves["tf_mse"][0] if curves["tf_mse"] else None,
        }
        out["floors"]["copy_last"][context] = {"mse": curves["copy_mse"]}
        out["floors"]["mean_frame"][context] = {"mse": curves["mean_mse"]}
        out["headline"][context] = {
            "mse_5_18": headline_mean(curves["ar_mse"], 5, 18),
            "ssim_5_18": headline_mean(curves["ar_ssim"], 5, 18),
        }
        out["horizons"][context] = {}
        n_steps = len(curves["ar_mse"])
        for h in horizons:
            idx = min(int(h), n_steps) - 1
            if idx < 0:
                continue
            out["horizons"][context][str(h)] = {
                "ar_mse": curves["ar_mse"][idx],
                "tf_mse": curves["tf_mse"][idx],
                "copy_mse": curves["copy_mse"][idx],
                "mean_mse": curves["mean_mse"][idx],
                "ar_ssim": curves["ar_ssim"][idx],
                "ar_latent_cos": curves["ar_latent_cos"][idx],
            }
    out["ar_first_vs_tf"] = {
        c: {
            "tf": out["tf"][c]["first_step_mse"],
            "ar": out["ar"][c]["mse"][0] if out["ar"][c]["mse"] else None,
        }
        for c in out["ar"]
    }
    return out
