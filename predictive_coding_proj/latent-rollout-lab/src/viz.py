from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_rollout_panel(true_frames, pred_frames, context, path, n_show=None):
    true_frames = true_frames.detach().cpu()
    pred_frames = pred_frames.detach().cpu()
    t_show = true_frames.shape[0] if n_show is None else min(n_show, true_frames.shape[0])
    shown_pred = torch.cat([true_frames[:context], pred_frames], dim=0)[:t_show]
    fig, axes = plt.subplots(2, t_show, figsize=(1.4 * t_show, 3.2))
    if t_show == 1:
        axes = np.array(axes).reshape(2, 1)
    for i in range(t_show):
        axes[0, i].imshow(true_frames[i, 0], cmap="gray", vmin=-0.5, vmax=0.5)
        axes[0, i].set_title(f"true t={i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(shown_pred[i, 0], cmap="gray", vmin=-0.5, vmax=0.5)
        label = "ctx" if i < context else "pred"
        axes[1, i].set_title(f"{label} t={i}")
        axes[1, i].axis("off")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_recon_panel(frames, recons, path, n_show=8):
    frames = frames.detach().cpu()
    recons = recons.detach().cpu()
    n_show = min(n_show, frames.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(1.5 * n_show, 3))
    if n_show == 1:
        axes = np.array(axes).reshape(2, 1)
    for i in range(n_show):
        axes[0, i].imshow(frames[i, 0], cmap="gray", vmin=-0.5, vmax=0.5)
        axes[0, i].set_title(f"True {i + 1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i, 0], cmap="gray", vmin=-0.5, vmax=0.5)
        axes[1, i].set_title(f"Recon {i + 1}")
        axes[1, i].axis("off")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_step_curves(runs, metric_key, context, path, ylabel=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, metrics in runs.items():
        curve = metrics.get("ar", {}).get(str(context), {}).get(metric_key)
        if curve is None:
            continue
        ax.plot(range(len(curve)), curve, label=name)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(f"AR {metric_key} (context={context})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
