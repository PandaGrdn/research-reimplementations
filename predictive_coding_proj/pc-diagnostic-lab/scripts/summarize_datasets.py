#!/usr/bin/env python3
"""One table per claim, one column per dataset, from finished runs.

    python scripts/summarize_datasets.py                       # default tags below
    python scripts/summarize_datasets.py --tags farmshare farmshare_bb farmshare_kth
    python scripts/summarize_datasets.py --json summary.json   # also write the numbers

A tag is the run directory name under results/<experiment>/ (the TAG the
FarmShare launcher used). Missing runs show as "-". Every number is read from
metrics.json; nothing is recomputed here except the Moving MNIST blank floor
for runs made before the rollout evaluation reported it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TAGS = ["farmshare", "farmshare_bb", "farmshare_kth"]
LABELS = {"farmshare": "Moving MNIST", "farmshare_bb": "Bouncing balls", "farmshare_kth": "KTH"}


def load(exp, tag):
    p = ROOT / "results" / exp / tag / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def c1_row(m):
    if m is None:
        return {}
    h = m["split_points"][str(m["headline_split"])]
    return {
        "rollout MSE": h["long_mse_mean"],
        "  seed std": h["long_mse_std"],
        "copy-last floor": h["copy_last_long_mse_mean"],
        "mean-frame floor": h["mean_frame_long_mse_mean"],
        "blank floor": h.get("blank_long_mse_mean"),
        "brightness kept": h.get("mass_ratio_long_mean"),
        "pixel corr": h.get("corr_long_mean"),
        "copy-last pixel corr": h.get("copy_last_corr_long_mean"),
        "teacher-forced MSE": m.get("tf_mse_mean"),
    }


def c2_rows(m):
    if m is None:
        return {}
    out = {}
    for c in m["cells"]:
        out[f"{c['name']} MSE"] = c["long_mse_mean"]
        if c.get("corr_long_mean") is not None:
            out[f"{c['name']} pixel corr"] = c["corr_long_mean"]
    return out


def c8_rows(m):
    if m is None:
        return {}
    out = {}
    for arm, v in m["arms"].items():
        out[f"{arm} rollout MSE"] = v.get("long_mse_mean")
        out[f"{arm} brightness kept"] = v.get("mass_ratio_long_mean")
        out[f"{arm} pixel corr"] = v.get("corr_long_mean")
    return out


def c13_rows(m, prefix):
    if m is None:
        return {}
    out = {}
    for kind, mv in m["models"].items():
        for arm in ("none", "consistency_top_down", "consistency_joint"):
            a = mv["arms"].get(arm)
            if a is None:
                continue
            out[f"{prefix}{kind} {arm} MSE"] = a["post_split"]["pixel_mse"]["mean"]
            if a.get("beat_none_count") is not None:
                out[f"{prefix}{kind} {arm} beats none"] = f"{a['beat_none_count']}/{a['beat_none_total']}"
    return out


def gather(tag):
    rows = {}
    rows["C1 rollout"] = c1_row(load("c1_rollout_collapse", tag))
    rows["C2 mitigations"] = c2_rows(load("c2_mitigation_grid", tag))
    c3 = load("c3_copy_detection", tag)
    rows["C3 copy bias"] = {} if c3 is None else {
        "final motion gap": c3["final_motion_gap_mean"],
        "final delta ratio": c3["final_delta_ratio_mean"],
    }
    c5 = load("c5_latent_smoothness", tag)
    rows["C5 smoothness"] = {} if c5 is None else {
        "code/pixel consec-vs-unrelated ratio": c5.get("code_over_pixel_frac_rel"),
        "code/pixel consecutive cosine": c5.get("code_over_pixel_cons_cos"),
    }
    c6 = load("c6_noise_floor", tag)
    rows["C6 settle noise"] = {} if c6 is None else {
        f"noise share of target ({c6['headline_protocol']})": c6["protocols"][c6["headline_protocol"]]["noise_share"],
        "noise share (cold, independent settles)": c6["protocols"]["cold_independent"]["noise_share"],
    }
    c9 = load("c9_predictability", tag)
    if c9 is not None:
        pm = c9.get("primary_model", "conv")
        pr = c9["predictability"][pm]
        rows["C9 predictability"] = {
            f"code R2 vs copy-last ({pm})": pr["code"]["r2_vs_copy_last_mean"],
            f"pixel R2 vs copy-last ({pm})": pr["pixel"]["r2_vs_copy_last_mean"],
            "exact-shift code cosine": c9["translation"]["exact_shift_cos_mean"],
            "aliased-shift code cosine": c9["translation"]["aliased_shift_cos_mean"],
        }
    else:
        rows["C9 predictability"] = {}
    rows["C8 inference procedure"] = c8_rows(load("c8_amortized_contrast", tag))
    c11 = load("c11_offline_gru", tag)
    rows["C11 offline GRU"] = {} if c11 is None else {
        "independent R2": c11["independent"]["best_r2_mean"],
        "coupled R2": c11["coupled"]["best_r2_mean"],
    }
    c12 = load("c12_energy_fade", tag)
    rows["C12 inter-layer error"] = {} if c12 is None else {
        "post-handoff e1 / true": c12["models"]["independent"].get("e1_ratio_post"),
        "pixel-energy spike from mean-centering": c12["models"]["independent"].get("dc_artifact_share_post"),
    }
    rows["C13 consistency settle"] = {
        **c13_rows(load("c13_consistency_rollout", tag), "offline "),
        **c13_rows(load("c13_consistency_rollout", f"{tag}_on_c1"), "C1 model "),
    }
    return rows


def dictionary_conflicts(tag):
    """Every experiment under one tag should have used the same dictionary
    weights for a given seed. Returns {seed: {fingerprint: [experiments]}} for
    seeds where they disagree."""
    seen = {}
    for exp_dir in sorted((ROOT / "results").glob(f"*/{tag}")) + sorted((ROOT / "results").glob(f"*/{tag}_on_c1")):
        fps = {}
        for name in ("env.json", "metrics.json"):
            f = exp_dir / name
            if f.exists():
                fps.update(json.loads(f.read_text()).get("dictionary_fingerprints") or {})
        for seed, fp in fps.items():
            seen.setdefault(str(seed), {}).setdefault(fp, []).append(exp_dir.parent.name)
    return {s: v for s, v in seen.items() if len(v) > 1}


def blank_floor_mnist():
    """Blank floor for Moving MNIST runs made before evaluation reported it."""
    import torch

    from src.config import resolve_config
    from src.experiment import load_data

    cfg = resolve_config(overlays=[ROOT / "configs" / "reporting.yaml"])
    vals = []
    for seed in (0, 1, 2):
        _, _, test, _ = load_data(cfg, ROOT, seed)
        sp = int(cfg["eval"]["headline_split"])
        x = torch.stack([s[sp:] for s in test[: cfg["eval"]["n_rollout_sequences"]]])
        vals.append(float((x ** 2).mean()))
    return sum(vals) / len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = {t: gather(t) for t in args.tags}
    if "farmshare" in data and data["farmshare"]["C1 rollout"] and data["farmshare"]["C1 rollout"].get("blank floor") is None:
        data["farmshare"]["C1 rollout"]["blank floor"] = blank_floor_mnist()
    names = [LABELS.get(t, t) for t in args.tags]
    for section in data[args.tags[0]]:
        keys = []
        for t in args.tags:
            keys += [k for k in data[t][section] if k not in keys]
        if not keys:
            continue
        print(f"\n## {section}")
        width = max(len(k) for k in keys) + 2
        print("".ljust(width) + "".join(n.rjust(16) for n in names))
        for k in keys:
            print(k.ljust(width) + "".join(fmt(data[t][section].get(k)).rjust(16) for t in args.tags))
    for t in args.tags:
        clash = dictionary_conflicts(t)
        if clash:
            print(f"\nWARNING {LABELS.get(t, t)}: experiments used different dictionaries for the same seed: {clash}")
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=1, default=str))


if __name__ == "__main__":
    main()
