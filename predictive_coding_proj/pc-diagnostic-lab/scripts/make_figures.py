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

    # Floors (copy-last / mean-frame) for the headline split only, so the
    # figure answers "collapse relative to what" without cluttering every split.
    headline_sp = str(m.get("headline_split", ""))
    h = splits.get(headline_sp)
    if h is not None:
        cl = h.get("copy_last_mse_per_frame_mean")
        mf = h.get("mean_frame_mse_per_frame_mean")
        if cl:
            ax.plot(np.arange(1, len(cl) + 1), cl, color="k", ls="--", lw=1.3,
                     label=f"copy-last floor (split={headline_sp})")
        if mf:
            ax.plot(np.arange(1, len(mf) + 1), mf, color="k", ls=":", lw=1.3,
                     label=f"mean-frame floor (split={headline_sp})")

    ax.set_xlabel("frame index (pred for t)")
    ax.set_ylabel("pixel MSE")
    ax.set_title("C1  long-horizon collapse at the handoff")
    ax.legend(fontsize=7)
    save(fig, out, "fig_c1")


def fig_c2(m, out):
    cells = m["cells"]
    names = [c["name"] for c in cells]
    y = np.asarray([c["long_mse_mean"] for c in cells])
    e = np.asarray([c["long_mse_std"] for c in cells])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(names))
    ax.bar(x, y, yerr=e, capsize=3, color="#4c72b0", alpha=0.9)

    cl = next((c.get("copy_last_long_mse_mean") for c in cells if c.get("copy_last_long_mse_mean") is not None), None)
    mf = next((c.get("mean_frame_long_mse_mean") for c in cells if c.get("mean_frame_long_mse_mean") is not None), None)
    if cl is not None:
        ax.axhline(cl, color="k", ls="--", lw=1.2, label=f"copy-last floor ({cl:.3f})")
    if mf is not None:
        ax.axhline(mf, color="k", ls=":", lw=1.2, label=f"mean-frame floor ({mf:.3f})")
    if cl is not None or mf is not None:
        ax.legend(fontsize=8)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))

    dvin = m.get("dist_vs_init_noise") or []
    nonzero = [d for d in dvin if d["init_noise"] > 0]
    zero = [d for d in dvin if d["init_noise"] == 0]
    if nonzero:
        x = [d["init_noise"] for d in nonzero]
        y = [d["abs_mean"] for d in nonzero]
        ax1.plot(x, y, marker="o", color="#4c72b0")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    slope = m.get("null_space_slope")
    slope_txt = f"{slope:.2f}" if isinstance(slope, (int, float)) and slope == slope else "n/a"  # nan/None check
    ax1.set_xlabel("init_noise")
    ax1.set_ylabel("‖r1 − r2‖ (abs dist)")
    ax1.set_title(f"C4  null-space floor vs init_noise (slope={slope_txt})")
    if zero:
        ax1.annotate(
            f"init_noise=0 →\nabs={zero[0]['abs_mean']:.3f} (exact)\ncos={zero[0]['cos_mean']:.3f}",
            xy=(0.04, 0.92),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.5"),
        )

    dvi = m.get("dist_vs_iters") or []
    xs = [d["iters"] for d in dvi]
    cos = np.asarray([d["cos_mean"] for d in dvi])
    energy = np.asarray([max(d["energy_mean"], 1e-12) for d in dvi])
    ax2.plot(xs, cos, marker="o", color="#4c72b0", label="cos(r1,r2)")
    ax2.set_xlabel("settle iterations")
    ax2.set_ylabel("same-frame cosine")
    ax2.set_ylim(0, 1.02)
    ax2b = ax2.twinx()
    ax2b.plot(xs, energy, marker="s", color="#dd8452", label="total energy")
    ax2b.set_yscale("log")
    ax2b.set_ylabel("PC energy (log)")
    ax2.set_title(f"C4  cos & energy vs iters (init_noise={m.get('headline_init_noise', 0.01)})")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    save(fig, out, "fig_c4")


def _violin(ax, data, positions, color):
    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(color)
        b.set_alpha(0.7)
    parts["cmeans"].set_color("k")


def fig_c5(m, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8))

    conditions = m.get("conditions") or {}
    order = [c for c in ("cold", "warm", "warm_zero_init", "pixel") if c in conditions]
    seeds = m.get("seeds") or []
    x = np.arange(len(order))
    means = np.asarray([conditions[c]["frac_rel"] for c in order])
    stds = []
    for c in order:
        vals = [s[c]["frac_rel"] for s in seeds if c in s]
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    stds = np.asarray(stds)
    colors = ["#4c72b0", "#55a868", "#8172b2", "#937860"]
    ax1.bar(x, means, yerr=stds, capsize=3, color=colors[: len(order)], alpha=0.9)
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=20, ha="right")
    ax1.set_ylabel("consec / unrelated rel-distance")
    ax1.set_title("C5  code smoothness vs pixel reference")

    sweep = m.get("iters_sweep") or []
    xs = [s["iters"] for s in sweep]
    y = np.asarray([s["cons_cos_mean"] for s in sweep])
    e = np.asarray([s["cons_cos_std"] for s in sweep])
    ax2.errorbar(xs, y, yerr=e, marker="o", capsize=3, color="#4c72b0", label="warm cons cos")
    if "pixel" in conditions:
        ax2.axhline(conditions["pixel"]["cons_cos"], color="k", ls="--", lw=1, label="pixel cons cos")
    ax2.set_xlabel("settle iterations")
    ax2.set_ylabel("consecutive cosine")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("warm-start code smoothness vs iters")
    ax2.legend()

    fig.suptitle("C5  latent smoothness vs pixel-space reference")
    save(fig, out, "fig_c5")


def fig_c6(m, out):
    protocols = m.get("protocols") or {}
    order = [p for p in ("cold_independent", "warm_independent_init", "pipeline") if p in protocols]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(order))
    noise_label_done = False
    remaining_label_done = False
    for xi, name in zip(x, order):
        d = protocols[name]
        noise_e = d["noise_energy"]
        target_e = d["target_energy"]
        if d["noise_over_target"] > 1.0:
            ax.bar([xi], [noise_e], color="#c44e52", label="noise energy" if not noise_label_done else None)
            noise_label_done = True
            ax.text(xi, noise_e * 1.02, "inconsistent", ha="center", va="bottom", fontsize=8, color="#c44e52")
        else:
            remaining = max(target_e - noise_e, 0.0)
            ax.bar([xi], [noise_e], color="#c44e52", label="noise energy" if not noise_label_done else None)
            noise_label_done = True
            ax.bar(
                [xi], [remaining], bottom=[noise_e],
                color="#55a868", label="remaining target energy" if not remaining_label_done else None,
            )
            remaining_label_done = True
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("energy (‖·‖²)")
    headline = m.get("headline_protocol", "warm_independent_init")
    hp = protocols.get(headline, {})
    ax.set_title(f"C6  noise vs target energy (headline: {headline}, noise/target={hp.get('noise_over_target', float('nan')):.2f})")
    ax.legend()
    save(fig, out, "fig_c6")


def fig_c7(m, out):
    """C7 | λ_slow × unroll_k. unroll_k=0 has no gradient path from the
    slowness term to the dictionary (settle detaches every step); unroll_k>0
    does (settle_with_temporal_prior_unrolled). no_pull isolates whether the
    dictionary itself changed shape; with_pull (dashed) is what the actual
    train/eval pipeline sees.
    """
    sweep = m.get("sweep") or []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    cmap = plt.get_cmap("tab10")

    if sweep and "unroll_k" in sweep[0]:
        ks = sorted(set(s["unroll_k"] for s in sweep))
        for i, k in enumerate(ks):
            rows = sorted([s for s in sweep if s["unroll_k"] == k], key=lambda s: s["lambda_slow"])
            x = [r["lambda_slow"] for r in rows]
            y_np = np.asarray([r.get("cons_cos_no_pull", r.get("cons_cos_mean")) for r in rows])
            e_np = np.asarray([r.get("cons_cos_no_pull_std", r.get("cons_cos_std", 0.0)) for r in rows])
            color = cmap(i)
            ax1.errorbar(x, y_np, yerr=e_np, marker="o", capsize=3, color=color,
                         label=f"no_pull  k={k}  {'(no grad path)' if k == 0 else '(grad path)'}")
            if "cons_cos_with_pull" in rows[0]:
                y_wp = np.asarray([r["cons_cos_with_pull"] for r in rows])
                ax1.plot(x, y_wp, marker="s", ls="--", color=color, alpha=0.8, label=f"with_pull  k={k}")

            drift = [r.get("dict_drift") for r in rows]
            if any(d is not None for d in drift):
                ax2.plot(x, drift, marker="o", color=color, label=f"k={k}")
        ax1.set_xlabel("λ_slow")
        ax1.set_ylabel("consecutive cosine")
        ax1.set_ylim(0, 1.02)
        ax1.set_title("smoothness: no_pull vs with_pull")
        ax1.legend(fontsize=6)
        ax2.set_xlabel("λ_slow")
        ax2.set_ylabel("dictionary drift (rel. Frobenius)")
        ax2.set_title("did the dictionary change?")
        ax2.legend(fontsize=7)
    else:
        # backward-compatible fallback for the pre-rework schema (no unroll_k).
        x = [s["lambda_slow"] for s in sweep]
        y = np.asarray([s.get("cons_cos_mean", s.get("cons_cos_no_pull")) for s in sweep])
        e = np.asarray([s.get("cons_cos_std", s.get("cons_cos_no_pull_std", 0.0)) for s in sweep])
        ax1.errorbar(x, y, yerr=e, marker="o", capsize=3, color="#4c72b0")
        ax1.set_xlabel("λ_slow")
        ax1.set_ylabel("consecutive cosine")
        ax1.set_ylim(0, 1.02)
        ax2.axis("off")

    fig.suptitle("C7  slowness × gradient path to the dictionary")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out, "fig_c7")


def fig_ladder(m, out):
    """Grouped bars per ablation arm: long_mse/copy-last ratio, frac_rel code
    vs pixel, R² code vs pixel. Panels with no data anywhere are skipped.
    """
    rows = m.get("rows") or []
    if not rows:
        return
    arms = [r["arm"] for r in rows]
    x = np.arange(len(arms))
    w = 0.35

    def _has(key_path):
        return any(_dig(r, key_path) is not None for r in rows)

    def _dig(r, key_path):
        cur = r
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    panels = []
    if _has(["c1", "long_mse"]) or _has(["c1", "copy_last_floor"]):
        panels.append("c1")
    if _has(["c5", "warm_frac_rel"]) or _has(["c5", "pixel_frac_rel"]):
        panels.append("c5")
    if _has(["c9", "code_r2"]) or _has(["c9", "pixel_r2"]):
        panels.append("c9")
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "c1":
            long_mse = np.asarray([_dig(r, ["c1", "long_mse"]) or 0.0 for r in rows])
            floor = np.asarray([_dig(r, ["c1", "copy_last_floor"]) or 0.0 for r in rows])
            ax.bar(x - w / 2, long_mse, width=w, color="#4c72b0", label="long MSE")
            ax.bar(x + w / 2, floor, width=w, color="#c44e52", label="copy-last floor")
            ax.set_ylabel("MSE")
            ax.set_title("C1  long-horizon MSE vs floor")
            ax.legend(fontsize=7)
        elif panel == "c5":
            warm = np.asarray([_dig(r, ["c5", "warm_frac_rel"]) or 0.0 for r in rows])
            pixel = np.asarray([_dig(r, ["c5", "pixel_frac_rel"]) or 0.0 for r in rows])
            ax.bar(x - w / 2, warm, width=w, color="#55a868", label="code (warm)")
            ax.bar(x + w / 2, pixel, width=w, color="#8172b2", label="pixel")
            ax.set_ylabel("consec/unrelated rel-distance")
            ax.set_title("C5  code vs pixel smoothness")
            ax.legend(fontsize=7)
        elif panel == "c9":
            code = np.asarray([_dig(r, ["c9", "code_r2"]) or 0.0 for r in rows])
            pixel = np.asarray([_dig(r, ["c9", "pixel_r2"]) or 0.0 for r in rows])
            ax.bar(x - w / 2, code, width=w, color="#4c72b0", label="code")
            ax.bar(x + w / 2, pixel, width=w, color="#dd8452", label="pixel")
            ax.axhline(0.0, color="k", ls="--", lw=0.8)
            ax.set_ylabel("R² vs copy-last")
            ax.set_title("C9  predictability")
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=25, ha="right")

    fig.suptitle("Ablation ladder: one variable at a time vs baseline")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out, "fig_ladder")


_ARM_ORDER_C8 = ["iterative", "amortized_init_settle", "amortized"]
_ARM_LABEL_C8 = {"iterative": "iterative", "amortized_init_settle": "init+settle", "amortized": "amortized"}


def fig_c8(m, out):
    """C8 ladder: same dictionary / temporal loop / evaluators, only the
    inference procedure (encoder=None vs encoder+settle vs encoder-only) changes.
    """
    arms_dict = m["arms"]
    order = [a for a in _ARM_ORDER_C8 if a in arms_dict]
    labels = [_ARM_LABEL_C8[a] for a in order]
    x = np.arange(len(order))
    w = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 3.8))

    ax = axes[0]
    tf_vals = np.asarray([arms_dict[a]["tf_mse_mean"] for a in order])
    long_vals = np.asarray([arms_dict[a]["long_mse_mean"] for a in order])
    ax.bar(x - w / 2, tf_vals, width=w, label="tf MSE", color="#4c72b0")
    ax.bar(x + w / 2, long_vals, width=w, label="long MSE", color="#dd8452")
    cl_tf = arms_dict[order[0]].get("copy_last_mse_mean")
    cl_long = arms_dict[order[0]].get("copy_last_long_mse_mean")
    if cl_tf is not None:
        ax.axhline(cl_tf, color="#4c72b0", ls="--", lw=1, alpha=0.8, label="copy-last (tf)")
    if cl_long is not None:
        ax.axhline(cl_long, color="#dd8452", ls="--", lw=1, alpha=0.8, label="copy-last (long)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("tf / long MSE vs copy-last floor")
    ax.legend(fontsize=7)

    ax = axes[1]
    data = [arms_dict[a].get("cons_cos_list") or [0.0] for a in order]
    _violin(ax, data, list(range(1, len(order) + 1)), "#55a868")
    px = (m.get("pixel_reference") or {}).get("cons_cos_mean")
    if px is not None:
        ax.axhline(px, color="#c44e52", ls="--", lw=1, label=f"pixel ref {px:.2f}")
        ax.legend(fontsize=8)
    ax.set_xticks(list(range(1, len(order) + 1)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("consecutive cosine")
    ax.set_ylim(0, 1.02)
    ax.set_title("latent smoothness")

    ax = axes[2]
    recon_vals = np.asarray([arms_dict[a].get("recon_mse_mean", 0.0) for a in order])
    energy_vals = np.asarray([arms_dict[a].get("energy_mean", 0.0) for a in order])
    ax.bar(x - w / 2, recon_vals, width=w, label="recon MSE", color="#4c72b0")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, energy_vals, width=w, label="energy", color="#8172b2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("recon MSE")
    ax2.set_ylabel("settle energy")
    ax.set_title("dictionary fit")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    ax = axes[3]
    primary_model = m.get("predictability_primary_model", "conv")
    r2_vals = np.asarray([
        (arms_dict[a].get("predictability", {}).get(primary_model, {}).get("code", {}) or {}).get(
            "r2_vs_copy_last_mean", 0.0
        )
        for a in order
    ])
    r2_errs = np.asarray([
        (arms_dict[a].get("predictability", {}).get(primary_model, {}).get("code", {}) or {}).get(
            "r2_vs_copy_last_std", 0.0
        )
        for a in order
    ])
    ax.bar(x, r2_vals, width=0.5, yerr=r2_errs, capsize=3, color="#4c72b0")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    pix_pred = (m.get("pixel_reference") or {}).get("predictability", {}).get(primary_model, {}) or {}
    px_r2 = pix_pred.get("r2_vs_copy_last_mean")
    if px_r2 is not None:
        ax.axhline(px_r2, color="#c44e52", ls="--", lw=1, label=f"pixel ref {px_r2:.2f}")
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("R² vs copy-last")
    ax.set_title(f"predictability ({primary_model})")

    fig.suptitle("C8  the inference-procedure ladder")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, out, "fig_c8")


def fig_c0(m, out):
    """C0 isolation test: settle(true frame) vs settle(synthetic frame) at the
    handoff, paired bars for energy / iters / two-init consistency.
    """
    s1 = m["step1"]
    s2 = m.get("step2") or {}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    specs = [("total_energy", "settle energy"), ("iters", "iters to converge"), ("two_init_cos", "two-init cosine")]
    colors = ["#4c72b0", "#c44e52"]
    for ax, (key, title) in zip(axes, specs):
        means = [s1["true"][key]["mean"], s1["synthetic"][key]["mean"]]
        errs = [s1["true"][key]["std"], s1["synthetic"][key]["std"]]
        ax.bar([0, 1], means, yerr=errs, capsize=3, color=colors)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["true", "synthetic"])
        ax.set_title(title)
    note = (
        f"cos(true,syn)={s1['cross_cos']['mean']:.3f}  "
        f"pixel mse={s1['pixel_mse_true_vs_syn']['mean']:.4f}  "
        f"sat={s1['saturation']['mean']:.4f}"
    )
    if s2.get("cross_cos") is not None:
        note += f"\nstep 2 (one further):  cos(true,syn)={s2['cross_cos']['mean']:.3f}"
    fig.text(0.5, -0.04, note, ha="center", fontsize=9)
    fig.suptitle("C0  settle: true frame vs synthetic frame at the handoff")
    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    save(fig, out, "fig_c0")


def fig_c9(m, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    pred = m["predictability"]
    models = list(pred.keys())
    x = np.arange(len(models))
    width = 0.35
    code_y = np.asarray([pred[mo]["code"]["r2_vs_copy_last_mean"] for mo in models])
    code_e = np.asarray([pred[mo]["code"]["r2_vs_copy_last_std"] for mo in models])
    pix_y = np.asarray([pred[mo]["pixel"]["r2_vs_copy_last_mean"] for mo in models])
    pix_e = np.asarray([pred[mo]["pixel"]["r2_vs_copy_last_std"] for mo in models])
    ax.bar(x - width / 2, code_y, width, yerr=code_e, capsize=3, color="#4c72b0", label="code")
    ax.bar(x + width / 2, pix_y, width, yerr=pix_e, capsize=3, color="#dd8452", label="pixel")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("R² vs copy-last")
    ax.set_title("C9  held-out predictability")
    ax.legend()

    ax2 = axes[1]
    per_shift = m["translation"]["per_shift"]
    xs = np.arange(len(per_shift))
    colors = ["#4c72b0", "#55a868"]
    bw = 0.35
    n_layers = max((len(s["layers"]) for s in per_shift), default=0)
    for l in range(n_layers):
        offs = xs + (l - (n_layers - 1) / 2.0) * bw
        for xi, s in zip(offs, per_shift):
            layer = next((ly for ly in s["layers"] if ly["layer"] == l), None)
            if layer is None:
                continue
            aliased = layer.get("aliased", False)
            ax2.bar(
                xi,
                layer["cos_mean"],
                bw,
                color=colors[l % len(colors)],
                hatch="//" if aliased else None,
                edgecolor="k" if aliased else colors[l % len(colors)],
                alpha=0.9,
                label=f"layer {l}" if xi == offs[0] else None,
            )
    pixel_y = [s["pixel_cos_mean"] for s in per_shift]
    ax2.plot(xs, pixel_y, color="k", marker="o", ls="--", label="pixel ref")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"({s['dx']},{s['dy']})" for s in per_shift], rotation=30, ha="right")
    ax2.set_ylabel("cos(shifted code, rolled code)")
    ax2.set_title("translation consistency (solid=exact stride, hatched=aliased)")
    ax2.legend()

    fig.suptitle("C9  predictability & translation consistency")
    save(fig, out, "fig_c9")


def fig_c11(m, out):
    """C11 | left: test R² vs epoch for the independent vs coupled offline
    GRUs (mean±std across seeds), against the two ceiling probes (matched to
    C9's exact protocol vs minibatched full-scale). Right: teacher-forced
    inter-layer error e1 for independent / coupled / the true codes
    themselves — coupled is ~0 by construction (see src.temporal.
    CoupledTopDownRNN), so a log scale keeps it visible next to the others.

    Falls back to the old (pre-rework) single-curve schema when metrics.json
    predates this rework (no "independent"/"coupled" keys), so the figure
    still renders instead of crashing on a stale run.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    indep = m.get("independent")
    coupled = m.get("coupled")

    if indep is not None and coupled is not None:
        for d, color, label in ((indep, "#4c72b0", "independent"), (coupled, "#c44e52", "coupled")):
            y = np.asarray(d.get("history_mean") or [])
            e = np.asarray(d.get("history_std") or [0.0] * len(y))
            if len(y):
                x = np.arange(len(y))
                ax1.plot(x, y, marker="o", ms=3, color=color, label=label)
                ax1.fill_between(x, y - e, y + e, color=color, alpha=0.15)

        matched = m.get("probe_matched_r2_mean")
        full = m.get("probe_full_best_r2_mean")
        if matched is not None:
            ax1.axhline(matched, color="k", ls="--", lw=1.2, label=f"probe matched (C9 ref) {matched:.2f}")
        if full is not None:
            ax1.axhline(full, color="#55a868", ls=":", lw=1.2, label=f"probe full {full:.2f}")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("R² vs copy-last")
        ax1.set_title("test R² (mean±std across seeds)")
        ax1.legend(fontsize=7)

        names = ["independent", "coupled", "true codes"]
        vals = [indep.get("e1_mean", 0.0), coupled.get("e1_mean", 0.0), m.get("true_e1_mean", 0.0)]
        errs = [indep.get("e1_std", 0.0), coupled.get("e1_std", 0.0), m.get("true_e1_std", 0.0)]
        colors = ["#4c72b0", "#c44e52", "#55a868"]
        x = np.arange(len(names))
        ax2.bar(x, [max(v, 1e-8) for v in vals], yerr=errs, capsize=3, color=colors, alpha=0.9)
        ax2.set_yscale("log")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=15, ha="right")
        ax2.set_ylabel("teacher-forced e1 (log)")
        ax2.set_title("inter-layer consistency error")
    else:
        # backward-compatible fallback for the pre-rework single-model schema.
        hist = m.get("history") or []
        if hist:
            xs = [h["epoch"] for h in hist]
            ys = [h["test_r2"] for h in hist]
            ax1.plot(xs, ys, marker="o", color="#4c72b0", label="offline GRU")
        target = m.get("target_r2")
        if target is not None:
            ax1.axhline(target, color="k", ls="--", lw=1.2, label=f"target {target:.2f}")
        avail = m.get("available_r2")
        if avail is not None:
            ax1.axhline(avail, color="#c44e52", ls=":", lw=1.2, label=f"C9 available {avail:.2f}")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("R² vs copy-last")
        ax1.legend(fontsize=8)
        ax2.axis("off")

    fig.suptitle("C11  offline ConvGRU on eval-protocol codes: independent vs coupled top-down")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out, "fig_c11")


def fig_c12(m, out):
    """C12: (1) artificial-fade control (energy vs scale, no rollout) — the
    baseline every claim below is read against; (2) independent model's
    per-step e1_pred vs e1_true (the honest inter-layer signal) with
    dc_offset on a twin axis (the artifact, now visible instead of hidden in
    a total); (3) mass_pred / r_pred_norm vs step for independent vs coupled.
    """
    control = m.get("control") or {}
    models = m.get("models") or {}
    indep = models.get("independent") or {}
    coupled = models.get("coupled") or {}
    split = int(m.get("headline_split", 0) or 0)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8))

    # Panel 1: control — energy vs scale, per term.
    ax = axes[0]
    per_scale = control.get("per_scale") or {}
    scales = control.get("scales") or sorted((float(s) for s in per_scale), default=[])
    if scales:
        xs = np.asarray(scales)
        for key, color, label in (
            ("total", "#4c72b0", "total"),
            ("e0", "#c44e52", "e0"),
            ("e1", "#55a868", "e1"),
        ):
            ys = np.asarray([per_scale.get(str(s), {}).get(f"{key}_mean", np.nan) for s in scales])
            es = np.asarray([per_scale.get(str(s), {}).get(f"{key}_std", 0.0) for s in scales])
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=color, label=label)
    ax.set_xlabel("contrast scale (0 = blank)")
    ax.set_ylabel("PC energy")
    ax.set_title(f"control: energy vs contrast (blank_is_minimum={control.get('blank_is_minimum')})")
    ax.legend(fontsize=7)

    # Panel 2: independent model — e1_pred vs e1_true, dc_offset on twin axis.
    ax = axes[1]
    curves = indep.get("curves") or {}
    e1p = np.asarray(curves.get("pred_e1_mean") or [])
    e1p_sd = np.asarray(curves.get("pred_e1_std") or [0.0] * len(e1p))
    e1t = np.asarray(curves.get("true_e1_mean") or [])
    e1t_sd = np.asarray(curves.get("true_e1_std") or [0.0] * len(e1t))
    dc = np.asarray(curves.get("pred_dc_offset_mean") or [])
    n = max(len(e1p), len(e1t), len(dc), 1)
    x = np.arange(1, n + 1)
    if len(e1p):
        ax.plot(x[: len(e1p)], e1p, color="#c44e52", label="e1 pred")
        ax.fill_between(x[: len(e1p)], e1p - e1p_sd, e1p + e1p_sd, color="#c44e52", alpha=0.15)
    if len(e1t):
        ax.plot(x[: len(e1t)], e1t, color="#4c72b0", label="e1 true")
        ax.fill_between(x[: len(e1t)], e1t - e1t_sd, e1t + e1t_sd, color="#4c72b0", alpha=0.15)
    ax.axvline(split, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("frame t")
    ax.set_ylabel("inter-layer energy e1")
    ax2 = ax.twinx()
    if len(dc):
        ax2.plot(x[: len(dc)], dc, color="#8172b2", ls=":", label="dc_offset (pred)")
    ax2.set_ylabel("dc_offset (pred decode mean)")
    ax.set_title("independent: e1 pred vs true + DC artifact")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    # Panel 3: mass_pred and r_pred_norm vs step, independent vs coupled.
    ax = axes[2]
    ax2 = ax.twinx()
    for name, payload, color in (
        ("independent", indep, "#4c72b0"),
        ("coupled", coupled if not coupled.get("skipped") else None, "#dd8452"),
    ):
        if not payload:
            continue
        c = payload.get("curves") or {}
        mass_p = np.asarray(c.get("mass_pred_mean") or [])
        rnorm_p = np.asarray(c.get("r_pred_norm_mean") or [])
        nn = max(len(mass_p), len(rnorm_p), 1)
        xs = np.arange(1, nn + 1)
        if len(mass_p):
            ax.plot(xs[: len(mass_p)], mass_p, color=color, label=f"mass_pred ({name})")
        if len(rnorm_p):
            ax2.plot(xs[: len(rnorm_p)], rnorm_p, color=color, ls="--", label=f"‖r_pred‖ ({name})")
    ax.axvline(split, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("frame t")
    ax.set_ylabel("mass_pred (mean |pixel|)")
    ax2.set_ylabel("‖r_pred‖ (dashed)")
    ax.set_title("fade + latent-norm blowup: independent vs coupled")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="best")

    fig.suptitle("C12  per-term energy vs closed-loop fade, with artificial-fade control")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out, "fig_c12")


_ARM_LABEL_C13 = {
    "none": "none",
    "consistency_top_down": "cons_td",
    "consistency_bottom_up": "cons_bu",
    "consistency_joint": "cons_joint",
    "image_settle": "image",
    "oracle_image": "oracle",
}


def fig_c13(m, out):
    """C13: does settling the rolled-out code on the inter-layer consistency
    term alone (no image term) reduce closed-loop drift, against the
    pure-latent baseline (none) and the information-carrying upper reference
    (oracle_image). (1) post-split pixel MSE per arm, independent vs coupled,
    with the copy-last floor; (2) cos(r_used, r_true) vs step, all independent
    arms; (3) mass_pred vs step for none / cons_td / image / oracle.
    """
    models = m.get("models") or {}
    indep = (models.get("independent") or {}).get("arms") or {}
    coupled = (models.get("coupled") or {}).get("arms") or {}
    order = [a for a in _ARM_LABEL_C13 if a in indep]
    labels = [_ARM_LABEL_C13[a] for a in order]
    split = int(m.get("headline_split", 0) or 0)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.8))

    ax = axes[0]
    if order:
        x = np.arange(len(order))
        w = 0.35
        indep_vals = np.asarray([indep[a]["post_split"].get("pixel_mse", {}).get("mean", np.nan) for a in order])
        ax.bar(x - w / 2, indep_vals, width=w, label="independent", color="#4c72b0")
        if coupled:
            coupled_vals = np.asarray([
                coupled.get(a, {}).get("post_split", {}).get("pixel_mse", {}).get("mean", np.nan) for a in order
            ])
            ax.bar(x + w / 2, coupled_vals, width=w, label="coupled (e1≡0)", color="#dd8452")
        floor = indep.get("none", {}).get("post_split", {}).get("copy_last_mse", {}).get("mean")
        if floor is not None:
            ax.axhline(floor, color="k", ls="--", lw=1.2, label=f"copy-last floor {floor:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("post-split pixel MSE")
    ax.set_title("closed-loop pixel MSE by arm")
    ax.legend(fontsize=7)

    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(order):
        curve = np.asarray((indep[a].get("curves") or {}).get("cos_r_mean") or [])
        if len(curve):
            ax.plot(np.arange(1, len(curve) + 1), curve, color=cmap(i), label=_ARM_LABEL_C13[a])
    ax.axvline(split, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("frame t")
    ax.set_ylabel("cos(r_used, r_true)")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("latent cosine drift by arm (independent)")
    ax.legend(fontsize=6)

    ax = axes[2]
    focus = [a for a in ("none", "consistency_top_down", "image_settle", "oracle_image") if a in indep]
    for i, a in enumerate(focus):
        curve = np.asarray((indep[a].get("curves") or {}).get("mass_pred_mean") or [])
        if len(curve):
            ax.plot(np.arange(1, len(curve) + 1), curve, color=cmap(i), label=_ARM_LABEL_C13[a])
    if focus:
        mass_true = np.asarray((indep[focus[0]].get("curves") or {}).get("mass_true_mean") or [])
        if len(mass_true):
            ax.plot(np.arange(1, len(mass_true) + 1), mass_true, color="k", ls=":", lw=1.3, label="true")
    ax.axvline(split, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("frame t")
    ax.set_ylabel("mean |pixel|")
    ax.set_title("fade (mass_pred) by arm")
    ax.legend(fontsize=6)

    fig.suptitle("C13  consistency-only test-time settle of the closed-loop code")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, out, "fig_c13")


def fig_c10(m, out):
    """C10 writes the inspection grids itself; copy seq 0 into figures/fig_c10."""
    gallery = m.get("gallery") or []
    if not gallery:
        return
    src = ROOT / gallery[0]["path"]
    if not src.exists():
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src, out / "fig_c10.png")
    print(f"wrote {out / 'fig_c10'}")


FIGURES = {
    "c1_rollout_collapse": fig_c1,
    "c2_mitigation_grid": fig_c2,
    "c3_copy_detection": fig_c3,
    "c4_settle_determinism": fig_c4,
    "c5_latent_smoothness": fig_c5,
    "c6_noise_floor": fig_c6,
    "c7_slowness_sweep": fig_c7,
    "c8_amortized_contrast": fig_c8,
    "c9_predictability": fig_c9,
    "c0_isolation_test": fig_c0,
    "c10_rollout_gallery": fig_c10,
    "c11_offline_gru": fig_c11,
    "c12_energy_fade": fig_c12,
    "c13_consistency_rollout": fig_c13,
    "ladder": fig_ladder,
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
