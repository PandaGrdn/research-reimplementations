from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PUB = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_pub_style():
    plt.rcParams.update(PUB)


def _as_hw(frame):
    x = frame.detach().cpu()
    while x.ndim > 2:
        x = x.squeeze(0)
    return x.numpy()


def save_rollout_panel(true_frames, pred_frames, context, path, n_show=None):
    t_show = true_frames.shape[0] if n_show is None else min(n_show, true_frames.shape[0])
    fig, axes = plt.subplots(2, t_show, figsize=(1.4 * t_show, 3.2))
    if t_show == 1:
        axes = np.array(axes).reshape(2, 1)
    for i in range(t_show):
        axes[0, i].imshow(_as_hw(true_frames[i]), cmap="gray", vmin=-0.5, vmax=0.5)
        axes[0, i].set_title(f"true t={i + 1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(_as_hw(pred_frames[i]), cmap="gray", vmin=-0.5, vmax=0.5)
        label = "ctx" if (i + 1) < context else "pred"
        axes[1, i].set_title(f"{label} t={i + 1}")
        axes[1, i].axis("off")
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_rollout_inspection(true_frames, tf_frames, ar_frames, copy_last_frames, split_point, path, n_show=None):
    """Four-row visual: ground truth, teacher-forced, closed-loop AR, copy-last freeze.

    Column i is the prediction for sequence frame t=i+1. A red bar marks the
    first column at/after the handoff (split_point).
    """
    t_show = true_frames.shape[0] if n_show is None else min(n_show, true_frames.shape[0])
    rows = [
        ("ground truth", true_frames),
        ("teacher-forced", tf_frames),
        ("closed-loop AR", ar_frames),
        ("copy-last", copy_last_frames),
    ]
    fig, axes = plt.subplots(len(rows), t_show, figsize=(1.15 * t_show, 1.35 * len(rows) + 0.6))
    if t_show == 1:
        axes = np.array(axes).reshape(len(rows), 1)
    handoff_col = max(split_point - 1, 0)
    for r, (label, frames) in enumerate(rows):
        for i in range(t_show):
            ax = axes[r, i]
            ax.imshow(_as_hw(frames[i]), cmap="gray", vmin=-0.5, vmax=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0 if i == handoff_col else 0.3)
                spine.set_edgecolor("#c44e52" if i == handoff_col else "0.8")
            if r == 0:
                kind = "GT" if i < handoff_col else "AR"
                ax.set_title(f"t={i + 1}\n{kind}", fontsize=7)
            if i == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(f"handoff at t={split_point}  (red box = first closed-loop frame)", fontsize=10)
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_recon_panel(frame, recon, path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(_as_hw(frame), cmap="gray", vmin=-0.5, vmax=0.5)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(_as_hw(recon), cmap="gray", vmin=-0.5, vmax=0.5)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
