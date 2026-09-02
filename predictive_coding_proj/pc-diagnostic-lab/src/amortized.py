"""Minimal amortized encoder arm for C8 (single conv AE + slowness + TemporalConvRNN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import consecutive_pair_loader, frame_loader
from src.metrics import pair_stats, summarize_pair_lists
from src.temporal import _perturb_r, _update_temporal_params
from src.utils import clone_r


class SimpleAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=32, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, latent_channels, kernel_size, stride=stride, padding=padding)
        self.decoder = nn.ConvTranspose2d(latent_channels, in_channels, kernel_size, stride=stride, padding=padding)

    def encode(self, frames):
        lead = frames.shape[:-3]
        x = frames.reshape(-1, *frames.shape[-3:])
        r = F.relu(self.encoder(x))
        return r.view(*lead, *r.shape[1:])

    def decode(self, r):
        lead = r.shape[:-3]
        x = r.reshape(-1, *r.shape[-3:])
        frames = self.decoder(x)
        return frames.view(*lead, 1, *frames.shape[-2:])

    def forward(self, frames):
        return self.decode(self.encode(frames))


def build_ae(cfg):
    a = cfg["amortized"]
    return SimpleAE(
        latent_channels=a.get("latent_channels", 32),
        kernel_size=a.get("kernel_size", 4),
        stride=a.get("stride", 2),
        padding=a.get("padding", 1),
    )


def train_ae(ae, train_seqs, cfg, device, log=print):
    a = cfg["amortized"]
    optimizer = torch.optim.Adam(ae.parameters(), lr=a["lr"])
    lambda_slow = a.get("lambda_slow", 0.0)
    if lambda_slow > 0:
        loader = consecutive_pair_loader(train_seqs, a["batch_size"], shuffle=True, seed=cfg.get("seed", 0))
    else:
        loader = frame_loader(train_seqs, a["batch_size"], shuffle=True, seed=cfg.get("seed", 0))
    ae.train()
    for epoch in range(a["epochs"]):
        total, n = 0.0, 0
        for batch in loader:
            if lambda_slow > 0:
                I_t, I_n = batch[0].to(device), batch[1].to(device)
                r_t, r_n = ae.encode(I_t), ae.encode(I_n)
                recon = F.mse_loss(ae.decode(r_t), I_t) + F.mse_loss(ae.decode(r_n), I_n)
                slow = F.mse_loss(r_t, r_n)
                loss = recon + lambda_slow * slow
                b = I_t.shape[0]
            else:
                I = batch[0].to(device)
                loss = F.mse_loss(ae(I), I)
                b = I.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * b
            n += b
        log(f"[AE Epoch {epoch + 1}] loss: {total / max(n, 1):.6f}")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    return ae


@torch.no_grad()
def encode_as_r_list(ae, frame):
    x = frame.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    r = ae.encode(x)
    return [r]


@torch.no_grad()
def amortized_same_frame_cos(ae, frames, device):
    cos_list = []
    for I in frames:
        x = I.float()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(device)
        r1 = ae.encode(x)
        r2 = ae.encode(x)
        cos_list.append(pair_stats([r1], [r2])["cos"])
    return float(sum(cos_list) / max(len(cos_list), 1)), cos_list


@torch.no_grad()
def amortized_smoothness(ae, seqs, unrelated_frames, device, max_unrelated_pairs=200, generator=None):
    cons_cos, cons_rel, cons_abs = [], [], []
    for seq in seqs:
        seq = seq.to(device)
        rs = [encode_as_r_list(ae, seq[t]) for t in range(seq.shape[0])]
        for t in range(1, seq.shape[0]):
            s = pair_stats(rs[t], rs[t - 1])
            cons_cos.append(s["cos"])
            cons_rel.append(s["rel"])
            cons_abs.append(s["abs"])
    u_rs = [encode_as_r_list(ae, I.to(device)) for I in unrelated_frames]
    un_cos, un_rel, un_abs = [], [], []
    pairs = [(i, j) for i in range(len(u_rs)) for j in range(i + 1, len(u_rs))]
    if len(pairs) > max_unrelated_pairs:
        g = generator if generator is not None else torch.Generator(device="cpu")
        idx = torch.randperm(len(pairs), generator=g)[:max_unrelated_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    for i, j in pairs:
        s = pair_stats(u_rs[i], u_rs[j])
        un_cos.append(s["cos"])
        un_rel.append(s["rel"])
        un_abs.append(s["abs"])
    cons = summarize_pair_lists(cons_cos, cons_rel, cons_abs)
    un = summarize_pair_lists(un_cos, un_rel, un_abs)
    frac_rel = cons["rel"] / (un["rel"] + 1e-8)
    return {
        "cons_cos": cons["cos"],
        "cons_rel": cons["rel"],
        "cons_abs": cons["abs"],
        "un_cos": un["cos"],
        "un_rel": un["rel"],
        "un_abs": un["abs"],
        "frac_rel": frac_rel,
        "n_cons": len(cons_cos),
        "n_un": len(un_cos),
        "cons_cos_list": cons_cos,
        "un_cos_list": un_cos,
    }


def train_amortized_temporal(ae, temporal_nn, seqs, cfg, device, log=print):
    inf = cfg["inference"]
    tcfg = cfg["temporal"]
    a = cfg["amortized"]
    lr_u = a.get("temporal_lr", inf["lr_u"])
    lambda_u = inf["lambda_u"]
    delta_loss_weight = tcfg.get("delta_loss_weight", 100.0)
    ss_p = tcfg.get("ss_p", 0.0)
    r_noise_std = tcfg.get("r_noise_std", 0.0)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)
    epochs = a.get("temporal_epochs", tcfg["epochs"])

    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for seq in seqs:
            seq = seq.to(device)
            with torch.no_grad():
                r_seq = [encode_as_r_list(ae, seq[t]) for t in range(seq.shape[0])]
            hidden = temporal_nn.init_hidden(r_seq[0])
            r_prev = r_seq[0]
            for t in range(1, len(r_seq)):
                r_in = _perturb_r([ri.detach() for ri in r_prev], r_noise_std)
                h_in = [hi.detach() for hi in hidden]
                r_pred, hidden, deltas = temporal_nn(r_in, h_in)
                loss = 0.0
                for j in range(len(r_pred)):
                    delta_target = (r_seq[t][j] - r_in[j]).detach()
                    loss = loss + delta_loss_weight * F.smooth_l1_loss(deltas[j], delta_target, reduction="mean")
                    loss = loss + F.mse_loss(r_pred[j], r_seq[t][j].detach())
                temporal_nn.zero_grad()
                loss.backward()
                _update_temporal_params(temporal_nn, lr_u, lambda_u, max_grad_norm=max_grad_norm)
                epoch_loss += float(loss.detach())
                n += 1
                if torch.rand(()) < ss_p:
                    r_prev = clone_r(r_pred)
                else:
                    r_prev = r_seq[t]
                hidden = [hi.detach() for hi in hidden]
        log(f"[Amortized temporal epoch {epoch + 1}] loss: {epoch_loss / max(n, 1):.6f}")
    return temporal_nn


@torch.no_grad()
def amortized_short_rollout(ae, temporal_nn, seq, device, context=2):
    seq = seq.to(device)
    T = seq.shape[0]
    r_seq = [encode_as_r_list(ae, seq[t]) for t in range(T)]
    hidden = temporal_nn.init_hidden(r_seq[0])
    r_prev = r_seq[0]
    true_frames, pred_frames, mse_list = [], [], []
    for t in range(1, T):
        r_in = r_prev if t > context else r_seq[t - 1]
        r_pred, hidden, _ = temporal_nn(r_in, hidden)
        I_hat = ae.decode(r_pred[0])
        I_true = seq[t : t + 1]
        true_frames.append(I_true.detach())
        pred_frames.append(I_hat.detach())
        mse_list.append(F.mse_loss(I_hat, I_true).item())
        if t >= context:
            r_prev = clone_r(r_pred)
        else:
            r_prev = r_seq[t]
        hidden = [hi.detach() for hi in hidden]
    return {
        "mse": float(sum(mse_list) / max(len(mse_list), 1)),
        "mse_per_frame": mse_list,
        "true_frames": torch.cat(true_frames, dim=0),
        "pred_frames": torch.cat(pred_frames, dim=0),
        "context": context,
    }
