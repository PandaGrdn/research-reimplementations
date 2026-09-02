#!/usr/bin/env python3
"""Rebuild publication figures from results/*/latest/metrics.json (or newest timestamp)."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz import apply_pub_style


def latest_metrics(results, exp_name):
    d = Path(results) / exp_name
    if not d.exists():
        return None
    latest = d / "latest" / "metrics.json"
    if latest.exists():
        return json.loads(latest.read_text())
    runs = sorted(
        [p for p in d.iterdir() if p.is_dir() and (p / "metrics.json").exists()],
        key=lambda p: p.name,
    )
    if not runs:
        return None
    return json.loads((runs[-1] / "metrics.json").read_text())


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def fig_c1(m, out):
    splits = m["split_points"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cmap = plt.get_cmap("tab10")
    for i, (sp, payload) in enumerate(sorted(splits.items(), key=lambda kv: int(kv[0]))):
        y = np.asarray(payload["mse_per_frame_mean"])
        e = np.asarray(payload.get("mse_per_frame_std") or [0] * len(y))
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=cmap(i), label=f"split={sp}")
        ax.fill_between(x, y - e, y + e, color=cmap(i), alpha=0.15)
        ax.axvline(int(sp), color=cmap(i), ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("frame index (pred for t)")
    ax.set_ylabel("pixel MSE")
    ax.set_title("C1  long-horizon collapse at the handoff")
    ax.legend()
    save(fig, out, "fig_c1")


def fig_c2(m, out):
    cells = m["cells"]
    names = [c["name"] for c in cells]
    y = np.asarray([c["long_mse_mean"] for c in cells])
    e = np.asarray([c["long_mse_std"] for c in cells])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(names))
    ax.bar(x, y, yerr=e, capsize=3, color="#4c72b0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("long-horizon MSE")
    ax.set_title("C2  mitigations do not fix collapse")
    save(fig, out, "fig_c2")


def fig_c3(m, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    g = np.asarray(m["motion_gap_mean"])
    ge = np.asarray(m.get("motion_gap_std") or [0] * len(g))
    r = np.asarray(m["delta_ratio_mean"])
    re = np.asarray(m.get("delta_ratio_std") or [0] * len(r))
    x = np.arange(len(g))
    ax.plot(x, g, label="motion_gap", color="#4c72b0")
    ax.fill_between(x, g - ge, g + ge, color="#4c72b0", alpha=0.15)
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("motion_gap (pred vs t minus pred vs t−1)")
    ax2 = ax.twinx()
    ax2.plot(x, r, label="delta_ratio", color="#dd8452")
    ax2.fill_between(x, r - re, r + re, color="#dd8452", alpha=0.15)
    ax2.set_ylabel("‖δ‖ / ‖r_t − r_{t−1}‖")
    ax.set_title("C3  copy-last / identity plateau")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    save(fig, out, "fig_c3")


def fig_c4(m, out):
    curve = m.get("cos_vs_iters") or []
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = [c["iters"] for c in curve]
    y = np.asarray([c["cos_mean"] for c in curve])
    e = np.asarray([c["cos_std"] for c in curve])
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color="#4c72b0")
    if curve:
        ax.axhline(curve[-1]["cos_mean"], color="#c44e52", ls="--", lw=1, label="residual floor")
        ax.legend()
    ax.set_xlabel("settle iterations")
    ax.set_ylabel("same-frame cosine")
    ax.set_title("C4  settle determinism vs iterations")
    ax.set_ylim(0, 1.02)
    save(fig, out, "fig_c4")


def _violin(ax, data, positions, color):
    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(color)
        b.set_alpha(0.7)
    parts["cmeans"].set_color("k")


def fig_c5(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, key, title in ((axes[0], "cold", "cold-start"), (axes[1], "warm", "warm-start")):
        cons = m[key].get("cons_cos_list") or [0.0]
        un = m[key].get("un_cos_list") or [0.0]
        _violin(ax, [cons, un], [1, 2], "#4c72b0")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["consecutive", "unrelated"])
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.text(
            0.5,
            0.05,
            f"dist ratio {m[key]['frac_rel_mean']:.2f}",
            transform=ax.transAxes,
            ha="center",
        )
    axes[0].set_ylabel("cosine")
    fig.suptitle("C5  consecutive vs unrelated latent cosine")
    save(fig, out, "fig_c5")


def fig_c6(m, out):
    noise = m["noise_share_mean"]
    signal = m["predictable_fraction_mean"]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.bar([0], [noise], color="#c44e52", label="settle noise (2σ²)")
    ax.bar([0], [signal], bottom=[noise], color="#55a868", label="residual signal")
    ax.set_xticks([0])
    ax.set_xticklabels(["target ‖r_t − r_{t−1}‖²"])
    ax.set_ylabel("share of target energy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"C6  predictable fraction = {signal:.2f}")
    ax.legend()
    save(fig, out, "fig_c6")


def fig_c7(m, out):
    sweep = m["sweep"]
    x = [s["lambda_slow"] for s in sweep]
    y = np.asarray([s["cons_cos_mean"] for s in sweep])
    e = np.asarray([s["cons_cos_std"] for s in sweep])
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color="#4c72b0")
    ax.set_xlabel("λ_slow")
    ax.set_ylabel("consecutive cosine")
    ax.set_title("C7  slowness plateau (no gradient path to the dictionary)")
    ax.set_ylim(0, 1.02)
    save(fig, out, "fig_c7")


def fig_c8(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, key, title in (
        (axes[0], "iterative", "iterative settle"),
        (axes[1], "amortized", "amortized encoder"),
    ):
        cons = m[key].get("cons_cos_list") or [0.0]
        un = m[key].get("un_cos_list") or [0.0]
        _violin(ax, [cons, un], [1, 2], "#4c72b0" if key == "iterative" else "#55a868")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["consecutive", "unrelated"])
        extra = ""
        if key == "amortized":
            extra = f"\nsame-frame cos={m[key].get('same_frame_cos_mean', 1.0):.2f}"
        ax.set_title(title + extra)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("cosine")
    fig.suptitle("C8  iterative vs amortized latent smoothness")
    save(fig, out, "fig_c8")


FIGURES = {
    "c1_rollout_collapse": fig_c1,
    "c2_mitigation_grid": fig_c2,
    "c3_copy_detection": fig_c3,
    "c4_settle_determinism": fig_c4,
    "c5_latent_smoothness": fig_c5,
    "c6_noise_floor": fig_c6,
    "c7_slowness_sweep": fig_c7,
    "c8_amortized_contrast": fig_c8,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=str(ROOT / "figures"))
    args = p.parse_args()
    apply_pub_style()
    missing = []
    for exp, fn in FIGURES.items():
        m = latest_metrics(args.results, exp)
        if m is None:
            missing.append(exp)
            print(f"skip {exp}: no metrics.json")
            continue
        fn(m, args.out)
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
