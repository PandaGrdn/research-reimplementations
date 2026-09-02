"""Deconv dictionary: make_variables + spatial pretrain, ported from the notebooks."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import clone_r, to_device_r


def f_clamp(x):
    return torch.clamp(x, min=-1.0, max=1.0)


def make_variables(
    image,
    initial_r_size=64,
    num_channels=16,
    num_layers=2,
    kernel_size=4,
    padding=1,
    stride=2,
    init_noise=0.01,
    device="cpu",
):
    I = image
    if I.ndim == 4:
        previous_num_channels = I.shape[1]
    elif I.ndim == 3:
        previous_num_channels = I.shape[0]
    else:
        previous_num_channels = 1
    r = []
    deconvs = []
    out_channels = num_channels * 2
    for i in range(num_layers):
        r_size = initial_r_size // (2 ** (i + 1))
        r.append(torch.randn(1, out_channels, r_size, r_size, device=device) * init_noise)
        deconvs.append(
            nn.ConvTranspose2d(
                out_channels,
                previous_num_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            ).to(device)
        )
        previous_num_channels = out_channels
        out_channels = out_channels * 2
    return I, r, nn.ModuleList(deconvs)


def make_from_cfg(cfg, device, image=None):
    s = cfg["spatial"]
    if image is None:
        size = s.get("initial_r_size", 64)
        image = torch.zeros(1, 1, size, size, device=device)
    return make_variables(
        image,
        initial_r_size=s.get("initial_r_size", 64),
        num_channels=s.get("num_channels", 16),
        num_layers=s.get("num_layers", 2),
        kernel_size=s.get("kernel_size", 4),
        padding=s.get("padding", 1),
        stride=s.get("stride", 2),
        init_noise=s.get("init_noise", 0.01),
        device=device,
    )


def normalize_kernels(deconv):
    deconv.weight /= torch.linalg.vector_norm(deconv.weight, ord=2, dim=(1, 2, 3), keepdim=True) + 1e-8


def _prepare_image(I):
    I_curr = I.float()
    if I_curr.ndim == 3:
        I_curr = I_curr.unsqueeze(0)
    return I_curr - I_curr.mean()


def settle_spatial(I, r, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers, use_prior=True):
    """Inner-loop r update with no temporal prior (spatial pretrain / recon eval)."""
    I_curr = _prepare_image(I)
    e = [None] * num_layers
    r = clone_r(r)
    for _ in range(num_epochs_inner):
        e[0] = I_curr - f_clamp(deconvs[0](r[0]))
        for i in range(1, num_layers):
            e[i] = r[i - 1] - f_clamp(deconvs[i](r[i]))
        for i in range(num_layers):
            cauchy_prior = alpha * (2 * r[i] / (1 + r[i] ** 2)) if use_prior else 0.0
            bottom_up = F.conv2d(e[i], deconvs[i].weight, padding=deconvs[i].padding, stride=deconvs[i].stride)
            dr = (1.0 / sigma_2) * bottom_up - cauchy_prior
            if i < num_layers - 1:
                dr = dr - (1.0 / sigma_2) * e[i + 1]
            r[i] = (r[i] + lr_r * dr).detach()
            r[i].requires_grad = True
    return r, e, I_curr


def update_dictionary(I, r, deconvs, lambda_u, lr_u, sigma_2, num_layers):
    I_curr = _prepare_image(I) if not torch.is_tensor(I) or I.ndim < 4 else I
    e = [None] * num_layers
    for j in range(num_layers):
        if j > 0:
            e[j] = r[j - 1] - f_clamp(deconvs[j](r[j]))
        else:
            e[0] = I_curr - f_clamp(deconvs[0](r[0]))
        loss = (1.0 / (2 * sigma_2)) * torch.sum(e[j] ** 2)
        deconvs[j].zero_grad()
        loss.backward()
        with torch.no_grad():
            if deconvs[j].weight.grad is not None:
                deconvs[j].weight -= lr_u * (deconvs[j].weight.grad + lambda_u * deconvs[j].weight)
                normalize_kernels(deconvs[j])
                deconvs[j].weight.grad = None
    return e


def recon_mse(I, r, deconvs):
    I_curr = _prepare_image(I)
    with torch.no_grad():
        I_hat = f_clamp(deconvs[0](r[0]))
        mse = torch.mean((I_curr - I_hat) ** 2).item()
    return mse, I_curr, I_hat


def validate_hierarchical_spatial(I, r, num_layers, deconvs):
    I_curr = _prepare_image(I)
    I_hat_L0 = deconvs[0](r[0])
    h = r[num_layers - 1]
    for i in range(num_layers - 1, -1, -1):
        h = deconvs[i](h)
    I_hat_top_down = h
    mse_L0 = torch.mean((I_curr - I_hat_L0) ** 2)
    mse_top_down = torch.mean((I_curr - I_hat_top_down) ** 2)
    return {
        "mse_L0": float(mse_L0.detach()),
        "mse_top_down": float(mse_top_down.detach()),
        "I_hat_L0": I_hat_L0.detach(),
        "I_hat_top_down": I_hat_top_down.detach(),
    }


validate_heirarchical = validate_hierarchical_spatial


def pretrain_spatial(
    frames,
    r_init,
    deconvs,
    alpha,
    lambda_u,
    lr_r,
    lr_u,
    sigma_2,
    num_epochs,
    num_epochs_inner,
    num_layers,
    use_prior=True,
    log=print,
):
    """Dictionary learning on independent frames (temporal notebook pretrain_spatial)."""
    last_mse = None
    last_vis = None
    for epoch in range(num_epochs):
        for I in frames:
            r_curr, _, I_curr = settle_spatial(
                I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers, use_prior=use_prior
            )
            update_dictionary(I_curr, r_curr, deconvs, lambda_u, lr_u, sigma_2, num_layers)
        r_vis, _, I_vis = settle_spatial(
            frames[0], r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers, use_prior=use_prior
        )
        last_mse, I_vis, I_hat = recon_mse(I_vis, r_vis, deconvs)
        last_vis = (I_vis, I_hat)
        log(f"[Spatial Epoch {epoch:03d}]  Recon MSE: {last_mse:.6f}")
    return deconvs, last_mse, last_vis


def freeze_dictionary(deconvs):
    for d in deconvs:
        d.weight.requires_grad = False


def unfreeze_dictionary(deconvs):
    for d in deconvs:
        d.weight.requires_grad = True


def dictionary_key(cfg, seed=0):
    payload = {
        "spatial": cfg.get("spatial", {}),
        "inference": {
            k: cfg.get("inference", {}).get(k)
            for k in ("alpha", "lambda_u", "lr_r", "lr_u", "sigma_2", "use_prior")
        },
            "n_train": cfg.get("data", {}).get("n_train"),
            "pretrain_seed": seed,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _layer_spec(deconv):
    return {
        "in_channels": deconv.in_channels,
        "out_channels": deconv.out_channels,
        "kernel_size": tuple(deconv.kernel_size),
        "stride": tuple(deconv.stride),
        "padding": tuple(deconv.padding),
    }


def save_dictionary(path, deconvs, r_init, cfg, val_mse=None, seed=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "deconv_state": [d.state_dict() for d in deconvs],
            "deconv_spec": [_layer_spec(d) for d in deconvs],
            "r_init": [ri.detach().cpu() for ri in r_init],
            "config": cfg.get("spatial"),
            "val_mse": val_mse,
            "key": dictionary_key(cfg, seed=seed),
        },
        path,
    )


def load_dictionary(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    deconvs = nn.ModuleList()
    for spec, state in zip(ckpt["deconv_spec"], ckpt["deconv_state"]):
        layer = nn.ConvTranspose2d(
            spec["in_channels"],
            spec["out_channels"],
            kernel_size=spec["kernel_size"],
            stride=spec["stride"],
            padding=spec["padding"],
            bias=False,
        )
        layer.load_state_dict(state)
        deconvs.append(layer.to(device))
    r_init = to_device_r(ckpt["r_init"], device)
    return deconvs, r_init, ckpt.get("val_mse")
