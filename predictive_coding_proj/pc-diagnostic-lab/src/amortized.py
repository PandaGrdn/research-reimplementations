"""Amortized encoder arm for C8: one conv encoder onto the SAME frozen dictionary
used by the iterative settle, so the inference procedure is the only thing that
changes between arms (see experiments/c8_amortized_contrast.py).

Public surface:
  PCEncoder(r_init, ...)                         -- the encoder module
  build_encoder(cfg, r_init)                      -- construct from config
  train_encoder(encoder, deconvs, train_seqs, cfg, device, r_init=None, log=print)
  make_encoder_fn(encoder, device)                 -- encoder-only arm hook
  make_init_settle_fn(encoder, deconvs, cfg, refine_iters)  -- encoder+settle arm hook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import consecutive_pair_loader, frame_loader
from src.inference import _prepare_image, settle_grounded
from src.spatial_pc import f_clamp


class PCEncoder(nn.Module):
    """Cascading stride-2 conv encoder mirroring the frozen deconv dictionary.

    One trunk conv per layer (stride 2, ReLU) plus a linear (no activation) 1x1
    head per layer so the output can be negative like a settled code. Per-layer
    output shapes are read off `r_init` at construction time, not hardcoded:
    layer i gets `r_init[i].shape[1]` channels at `r_init[i].shape[-2:]` spatial
    size, which is exactly what the matching cascade of kernel=4/stride=2/pad=1
    convs produces starting from a `[1, in_channels, H, W]` frame (H, W the size
    used to build r_init).
    """

    def __init__(self, r_init, in_channels=1, kernel_size=4, stride=2, padding=1, hidden=64):
        super().__init__()
        self.num_layers = len(r_init)
        self.trunk = nn.ModuleList()
        self.heads = nn.ModuleList()
        prev_ch = in_channels
        for ri in r_init:
            out_ch = ri.shape[1]
            self.trunk.append(nn.Conv2d(prev_ch, hidden, kernel_size, stride=stride, padding=padding))
            self.heads.append(nn.Conv2d(hidden, out_ch, kernel_size=1))
            prev_ch = hidden

    def forward(self, x):
        codes = []
        h = x
        for conv, head in zip(self.trunk, self.heads):
            h = F.relu(conv(h))
            codes.append(head(h))
        return codes


def build_encoder(cfg, r_init):
    a = cfg["amortized"]
    s = cfg["spatial"]
    return PCEncoder(
        r_init,
        in_channels=1,
        kernel_size=a.get("kernel_size", s.get("kernel_size", 4)),
        stride=a.get("stride", s.get("stride", 2)),
        padding=a.get("padding", s.get("padding", 1)),
        hidden=a.get("hidden", 64),
    )


def _energy_loss(encoder, deconvs, I, sigma_2, alpha, use_prior):
    """The SAME PC energy settle minimises, through the frozen dictionary."""
    I_c = _prepare_image(I)
    r = encoder(I_c)
    num_layers = len(deconvs)
    e = [None] * num_layers
    e[0] = I_c - f_clamp(deconvs[0](r[0]))
    for i in range(1, num_layers):
        e[i] = r[i - 1] - f_clamp(deconvs[i](r[i]))
    loss = 0.0
    for i in range(num_layers):
        loss = loss + (1.0 / (2 * sigma_2)) * torch.mean(e[i] ** 2)
        if use_prior:
            loss = loss + alpha * torch.mean(torch.log(1 + r[i] ** 2))
    return loss, r


def _distill_loss(encoder, deconvs, I, r_init, cfg):
    inf = cfg["inference"]
    num_layers = len(deconvs)
    I_c = _prepare_image(I)
    with torch.no_grad():
        targets = [[] for _ in range(num_layers)]
        for b in range(I_c.shape[0]):
            r_b, _ = settle_grounded(
                I_c[b : b + 1],
                r_init,
                deconvs,
                inf["alpha"],
                inf["lr_r"],
                inf["sigma_2"],
                inf["num_epochs_inner"],
                num_layers,
                r_warm=None,
                init_noise=0.0,
                use_prior=inf.get("use_prior", True),
            )
            for i in range(num_layers):
                targets[i].append(r_b[i])
    r = encoder(I_c)
    loss = 0.0
    for i in range(num_layers):
        loss = loss + F.mse_loss(r[i], torch.cat(targets[i], dim=0))
    return loss, r


def train_encoder(encoder, deconvs, train_seqs, cfg, device, r_init=None, log=print):
    """Train PCEncoder to (by default) minimise the frozen dictionary's PC energy
    on shuffled training frames. Freezes the encoder in-place when done.
    """
    a = cfg["amortized"]
    inf = cfg["inference"]
    objective = a.get("objective", "energy")
    lambda_slow = a.get("lambda_slow", 0.0)
    if objective == "distill" and r_init is None:
        raise ValueError("train_encoder(objective='distill') requires r_init")
    if lambda_slow > 0:
        log(f"[amortized] lambda_slow={lambda_slow} > 0: slowness term is ON in encoder training")
        loader_fn = lambda: consecutive_pair_loader(train_seqs, a["batch_size"], shuffle=True, seed=cfg.get("seed", 0))
    else:
        loader_fn = lambda: frame_loader(train_seqs, a["batch_size"], shuffle=True, seed=cfg.get("seed", 0))

    def batch_loss(I):
        if objective == "distill":
            return _distill_loss(encoder, deconvs, I, r_init, cfg)
        return _energy_loss(encoder, deconvs, I, inf["sigma_2"], inf["alpha"], inf.get("use_prior", True))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=a["lr"])
    encoder.train()
    for epoch in range(a["epochs"]):
        total, n = 0.0, 0
        for batch in loader_fn():
            if lambda_slow > 0:
                I_t, I_n = batch[0].to(device), batch[1].to(device)
                loss_t, r_t = batch_loss(I_t)
                loss_n, r_n = batch_loss(I_n)
                slow = sum(F.mse_loss(rt, rn) for rt, rn in zip(r_t, r_n)) / len(r_t)
                loss = loss_t + loss_n + lambda_slow * slow
                b = I_t.shape[0]
            else:
                I = batch[0].to(device)
                loss, _ = batch_loss(I)
                b = I.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * b
            n += b
        log(f"[Encoder Epoch {epoch + 1}/{a['epochs']}] {objective} loss: {total / max(n, 1):.6f}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def make_encoder_fn(encoder, device):
    """Arm 3 (`amortized`): encoder only. Returns f(I[1,1,H,W]) -> list[Tensor],
    matching the `encoder=` hook of train_video_sequence / validate_hierarchical(_long).
    """
    encoder = encoder.to(device)
    encoder.eval()

    @torch.no_grad()
    def f(I):
        x = I.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        return [ri.detach() for ri in encoder(x)]

    return f


def make_init_settle_fn(encoder, deconvs, cfg, refine_iters):
    """Arm 2 (`amortized_init_settle`): encoder output used as the settle init,
    then `refine_iters` steps of the plain spatial settle (deterministic --
    `settle_grounded` never draws a random init when `r_warm` is given).
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    device = next(encoder.parameters()).device
    encoder.eval()

    @torch.no_grad()
    def f(I):
        x = I.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        r0 = [ri.detach() for ri in encoder(x)]
        r_refined, _ = settle_grounded(
            x,
            r0,
            deconvs,
            inf["alpha"],
            inf["lr_r"],
            inf["sigma_2"],
            refine_iters,
            num_layers,
            r_warm=r0,
            init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True),
        )
        return [ri.detach() for ri in r_refined]

    return f
