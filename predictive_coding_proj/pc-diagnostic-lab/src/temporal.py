"""ConvGRU temporal net + train_video_sequence, ported from the temporal notebook."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference import settle_with_temporal_prior
from src.metrics import delta_ratio
from src.spatial_pc import f_clamp, normalize_kernels
from src.utils import clone_r


class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        zr = torch.sigmoid(self.conv_zr(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class TemporalConvRNN(nn.Module):
    def __init__(self, r, delta_scale=0.1, delta_bounded=True):
        super().__init__()
        self.delta_scale = delta_scale
        self.delta_bounded = delta_bounded
        self.cells = nn.ModuleList()
        self.readouts = nn.ModuleList()
        for ri in r:
            ch = ri.shape[1]
            self.cells.append(ConvGRUCell(ch))
            readout = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)
            self.readouts.append(readout)

    def init_hidden(self, r_like):
        return [torch.zeros_like(ri) for ri in r_like]

    def forward(self, r_in, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(r_in)
        r_pred, h_new, deltas = [], [], []
        for cell, readout, x, h in zip(self.cells, self.readouts, r_in, hidden):
            h_t = cell(x, h)
            raw = readout(h_t)
            if self.delta_bounded:
                delta = self.delta_scale * torch.tanh(raw)
            else:
                delta = raw
            deltas.append(delta)
            r_pred.append(x + delta)
            h_new.append(h_t)
        return r_pred, h_new, deltas


TemporalNN = TemporalConvRNN


def _perturb_r(rs, noise_std):
    if noise_std is None or noise_std <= 0:
        return rs
    return [ri + torch.randn_like(ri) * noise_std for ri in rs]


def _update_temporal_params(temporal_nn, lr_u, lambda_u, max_grad_norm=1.0):
    torch.nn.utils.clip_grad_norm_(temporal_nn.parameters(), max_grad_norm)
    with torch.no_grad():
        for param in temporal_nn.parameters():
            if param.grad is not None:
                param -= lr_u * (param.grad + lambda_u * param)
                param.grad = None


def train_loop(
    I_curr,
    r_prev1,
    r_curr,
    alpha,
    lambda_u,
    lr_r,
    lr_u,
    sigma_2,
    num_epochs_outer,
    num_epochs_inner,
    num_layers,
    deconvs,
    temporal_nn,
    hidden=None,
    r_noise_std=0.0,
    lambda_slow=0.1,
    delta_loss_weight=20.0,
    delta_target_loss=True,
    use_prior=True,
    max_grad_norm=1.0,
):
    I_curr = I_curr.float()
    if I_curr.ndim == 3:
        I_curr = I_curr.unsqueeze(0)
    e_spatial = [None] * num_layers
    e_temporal = [None] * num_layers

    has_prev = r_prev1 is not None
    if r_prev1 is None:
        r_prev1 = [torch.zeros_like(ri) for ri in r_curr]
    else:
        r_prev1 = [ri.detach() for ri in r_prev1]

    if hidden is None:
        hidden = temporal_nn.init_hidden(r_curr)
    else:
        hidden = [hi.detach() for hi in hidden]

    h_new = hidden
    r_pred = None
    delta_stats = (0.0, 0.0)

    for out_epoch in range(num_epochs_outer):
        r_in = _perturb_r([ri.detach() for ri in r_prev1], r_noise_std)
        h_in = [hi.detach() for hi in hidden]
        r_pred, h_new, deltas = temporal_nn(r_in, h_in)

        r_curr, e_spatial, I_ready = settle_with_temporal_prior(
            I_curr,
            r_curr,
            r_pred,
            deconvs,
            alpha,
            lr_r,
            sigma_2,
            num_epochs_inner,
            num_layers,
            r_prev1=r_prev1 if has_prev else None,
            lambda_slow=lambda_slow,
            use_prior=use_prior,
        )

        e_spatial[0] = I_ready - f_clamp(deconvs[0](r_curr[0]))
        r_pred, h_new, deltas = temporal_nn(r_in, h_in)
        total_loss = 0

        for j in range(num_layers):
            if j > 0:
                e_spatial[j] = r_curr[j - 1] - f_clamp(deconvs[j](r_curr[j]))
            e_temporal[j] = r_curr[j] - r_pred[j]
            delta_target = (r_curr[j] - r_prev1[j]).detach()
            total_loss += (1.0 / (2 * sigma_2)) * torch.mean(e_spatial[j] ** 2)
            if delta_target_loss:
                total_loss += delta_loss_weight * F.smooth_l1_loss(deltas[j], delta_target, reduction="mean")
            if has_prev and lambda_slow > 0:
                total_loss += lambda_slow * torch.mean((r_curr[j] - r_prev1[j]) ** 2)
            deconvs[j].zero_grad()
        temporal_nn.zero_grad()
        total_loss.backward()

        with torch.no_grad():
            d_norm = sum(torch.norm(d).item() for d in deltas)
            a_norm = sum(torch.norm(r_curr[j] - r_prev1[j]).item() for j in range(num_layers))
            delta_stats = (d_norm, a_norm)
            for j in range(num_layers):
                if deconvs[j].weight.requires_grad and deconvs[j].weight.grad is not None:
                    deconvs[j].weight -= lr_u * (deconvs[j].weight.grad + lambda_u * deconvs[j].weight)
                    normalize_kernels(deconvs[j])
                    deconvs[j].weight.grad = None
        _update_temporal_params(temporal_nn, lr_u, lambda_u, max_grad_norm=max_grad_norm)

    return r_curr, r_pred, [hi.detach() for hi in h_new], delta_stats


def rollout_temporal_loss(
    r_in,
    hidden,
    r_targets,
    temporal_nn,
    sigma_2,
    lr_u,
    lambda_u,
    r_noise_std=0.0,
    delta_loss_weight=20.0,
    max_grad_norm=1.0,
):
    """k-step closed-loop ConvRNN rollout in r-space; backprop through the full unroll."""
    x = _perturb_r([ri.detach() for ri in r_in], r_noise_std)
    h = [hi.detach() for hi in hidden]
    loss = 0.0
    for target in r_targets:
        r_pred, h, deltas = temporal_nn(x, h)
        for j in range(len(r_pred)):
            delta_target = (target[j] - x[j]).detach()
            loss = loss + delta_loss_weight * F.smooth_l1_loss(deltas[j], delta_target, reduction="mean")
        x = r_pred
    temporal_nn.zero_grad()
    loss.backward()
    _update_temporal_params(temporal_nn, lr_u, lambda_u, max_grad_norm=max_grad_norm)
    return float(loss.detach())


def train_video_sequence(
    sequence,
    r_curr_init,
    alpha,
    lambda_u,
    lr_r,
    lr_u,
    sigma_2,
    num_epochs_outer,
    num_epochs_inner,
    num_layers,
    deconvs,
    temporal_nn,
    r_noise_std=0.0,
    ss_p=0.0,
    rollout_k=1,
    lambda_slow=0.1,
    delta_loss_weight=20.0,
    delta_target_loss=True,
    use_prior=True,
    max_grad_norm=1.0,
):
    r_prev1 = None
    hidden = None
    r_curr = r_curr_init
    r_sequence = []
    h_sequence = []
    delta_norms = []
    dr_norms = []

    for i, image in enumerate(sequence):
        I_curr = image.float()

        r_curr, r_pred, hidden, (d_norm, a_norm) = train_loop(
            I_curr,
            r_prev1,
            r_curr,
            alpha,
            lambda_u,
            lr_r,
            lr_u,
            sigma_2,
            num_epochs_outer,
            num_epochs_inner,
            num_layers,
            deconvs,
            temporal_nn,
            hidden=hidden,
            r_noise_std=r_noise_std,
            lambda_slow=lambda_slow,
            delta_loss_weight=delta_loss_weight,
            delta_target_loss=delta_target_loss,
            use_prior=use_prior,
            max_grad_norm=max_grad_norm,
        )

        if r_prev1 is not None and a_norm > 0:
            delta_norms.append(d_norm)
            dr_norms.append(a_norm)

        r_sequence.append(clone_r(r_curr))
        h_sequence.append([hi.detach().clone() for hi in hidden])

        if r_prev1 is not None and np.random.rand() < ss_p:
            r_prev1 = clone_r(r_pred)
        else:
            r_prev1 = [ri.detach() for ri in r_curr]

    if rollout_k > 1 and len(r_sequence) >= 1 + rollout_k:
        for t in range(1, len(r_sequence) - rollout_k + 1):
            rollout_temporal_loss(
                r_sequence[t - 1],
                h_sequence[t - 1],
                r_sequence[t : t + rollout_k],
                temporal_nn,
                sigma_2,
                lr_u,
                lambda_u,
                r_noise_std=r_noise_std,
                delta_loss_weight=delta_loss_weight,
                max_grad_norm=max_grad_norm,
            )

    ratio, mean_delta, mean_dr = delta_ratio(delta_norms, dr_norms)
    return r_sequence, {"delta_norm": mean_delta, "dr_norm": mean_dr, "ratio": ratio}


def inference_kwargs(cfg):
    inf = cfg["inference"]
    t = cfg["temporal"]
    return {
        "alpha": inf["alpha"],
        "lambda_u": inf["lambda_u"],
        "lr_r": inf["lr_r"],
        "lr_u": inf["lr_u"],
        "sigma_2": inf["sigma_2"],
        "num_epochs_outer": inf["num_epochs_outer"],
        "num_epochs_inner": inf["num_epochs_inner"],
        "num_layers": cfg["spatial"]["num_layers"],
        "r_noise_std": t.get("r_noise_std", 0.0),
        "ss_p": t.get("ss_p", 0.0),
        "rollout_k": t.get("rollout_k", 1),
        "lambda_slow": t.get("lambda_slow", 0.0),
        "delta_loss_weight": t.get("delta_loss_weight", 20.0),
        "delta_target_loss": t.get("delta_target_loss", True),
        "use_prior": inf.get("use_prior", True),
        "max_grad_norm": t.get("max_grad_norm", 1.0),
    }
