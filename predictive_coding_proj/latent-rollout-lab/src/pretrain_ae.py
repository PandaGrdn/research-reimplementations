from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import frame_loader
from src.models.autoencoder import build_ae
from src.utils import get_device
from src.viz import save_recon_panel


def train_ae(ae, train_seqs, val_seqs, cfg, device, log=print, fig_path=None):
    a = cfg["ae"]
    optimizer = torch.optim.Adam(ae.parameters(), lr=a["lr"])
    loader = frame_loader(train_seqs, a["batch_size"], shuffle=True, seed=cfg.get("seed", 0))
    ae.train()
    for epoch in range(a["epochs"]):
        total, n = 0.0, 0
        for (images,) in loader:
            images = images.to(device)
            hat = ae(images)
            loss = F.mse_loss(hat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * images.shape[0]
            n += images.shape[0]
        log(f"Epoch {epoch + 1} recon MSE: {total / max(n, 1):.6f}")
    val_mse, last_i, last_hat = eval_ae(ae, val_seqs, a["batch_size"], device)
    log(f"Val recon MSE: {val_mse:.6f}")
    if fig_path is not None and last_i is not None:
        save_recon_panel(last_i, last_hat, fig_path)
    return val_mse


@torch.no_grad()
def eval_ae(ae, seqs, batch_size, device):
    ae.eval()
    loader = frame_loader(seqs, batch_size, shuffle=False)
    total, n = 0.0, 0
    last_i, last_hat = None, None
    for (images,) in loader:
        images = images.to(device)
        hat = ae(images)
        total += F.mse_loss(hat, images).item() * images.shape[0]
        n += images.shape[0]
        last_i, last_hat = images, hat
    return total / max(n, 1), last_i, last_hat


def save_ae(path, ae, cfg, val_recon_mse):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": ae.state_dict(),
            "config": cfg["ae"],
            "val_recon_mse": val_recon_mse,
        },
        path,
    )


def load_ae(path, cfg, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    ae = build_ae(cfg).to(device)
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae, ckpt.get("val_recon_mse")
