#!/usr/bin/env python3
"""The ablation ladder: run a subset of experiments once per arm, then collect
one variable-at-a-time comparison table.

Each arm changes exactly one thing relative to `baseline` (see configs/ablation/
and README.md's "Target design" table). For each arm this script:

  1. runs the requested experiments (`--only`, default c1 c5 c9) with
     `--overlay configs/ablation/<arm>.yaml` (no overlay for baseline),
     passing `--smoke` through if given;
  2. finds each experiment's most recent run directory whose config.yaml
     records `arm: <arm>` (not just the newest run overall — another arm may
     have run more recently);
  3. pulls out a small, defensively-read set of headline numbers per claim;
  4. writes results/ladder/<stamp>/metrics.json (+ config.yaml, env.json, via
     the same finish_run() every other experiment uses) and prints a
     markdown table.

If an experiment script named in --only does not exist yet (e.g. this repo's
C9 lands after this script was written), that experiment is skipped for every
arm with a warning rather than failing the whole ladder.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import lab_root, resolve_config
from src.utils import finish_run, new_run_dir

EXP_MAP = {
    "c1": "c1_rollout_collapse",
    "c2": "c2_mitigation_grid",
    "c3": "c3_copy_detection",
    "c4": "c4_settle_determinism",
    "c5": "c5_latent_smoothness",
    "c6": "c6_noise_floor",
    "c7": "c7_slowness_sweep",
    "c8": "c8_amortized_contrast",
    "c9": "c9_predictability",
}

DEFAULT_ARMS = ["baseline", "zero_init", "complete_dict", "no_temporal_prior"]
DEFAULT_ONLY = ["c1", "c5", "c9"]


def parse_args():
    p = argparse.ArgumentParser(description="Run the ablation ladder and summarize")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--only", nargs="+", default=None, help="subset of c1..c9 (default: c1 c5 c9)")
    p.add_argument("--arms", nargs="+", default=None, help="override the arm list")
    p.add_argument("--skip-run", action="store_true", help="only collect + summarize existing results")
    return p.parse_args()


def arm_list(cfg, override):
    if override:
        return override
    return cfg.get("ladder", {}).get("arms", DEFAULT_ARMS)


def run_one(tag, arm, smoke, root):
    exp_name = EXP_MAP.get(tag)
    if exp_name is None:
        print(f"WARNING: unknown experiment tag '{tag}' — skipping")
        return False
    exp_file = ROOT / "experiments" / f"{exp_name}.py"
    if not exp_file.exists():
        print(f"WARNING: {exp_file.name} does not exist yet — skipping {tag} for all arms")
        return False
    cmd = [sys.executable, str(exp_file)]
    if smoke:
        cmd.append("--smoke")
    if arm != "baseline":
        overlay = ROOT / "configs" / "ablation" / f"{arm}.yaml"
        if not overlay.exists():
            print(f"WARNING: no overlay {overlay} for arm '{arm}' — skipping {tag}/{arm}")
            return False
        cmd += ["--overlay", str(overlay)]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return True


def find_latest_run_for_arm(exp_name, arm, root):
    """Newest results/<exp_name>/<run>/ whose config.yaml records arm==arm.

    Reads `arm` back out of each run's config.yaml rather than trusting the
    directory-name suffix (`new_run_dir` only appends `_<arm>` for non-baseline
    arms, and `latest` always points at whichever experiment ran most
    recently regardless of arm), so this is safe to call after several arms
    have run the same experiment out of order.
    """
    d = Path(root) / "results" / exp_name
    if not d.exists():
        return None
    candidates = []
    for p in sorted(d.iterdir()):
        if not p.is_dir() or p.name == "latest":
            continue
        cfg_path = p / "config.yaml"
        metrics_path = p / "metrics.json"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        try:
            run_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            continue
        if run_cfg.get("arm", "baseline") == arm:
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    latest = candidates[-1]
    try:
        return json.loads((latest / "metrics.json").read_text())
    except Exception:
        return None


def _get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_c1(m):
    if not m:
        return {}
    hs = str(m.get("headline_split", ""))
    splits = m.get("split_points", {}) or {}
    h = splits.get(hs) or (next(iter(splits.values())) if splits else {})
    return {
        "long_mse": h.get("long_mse_mean"),
        "copy_last_floor": h.get("copy_last_long_mse_mean"),
        "mean_frame_floor": h.get("mean_frame_long_mse_mean"),
        "tf_mse": m.get("tf_mse_mean"),
        "tf_copy_last": m.get("tf_copy_last_mean"),
    }


def extract_c5(m):
    if not m:
        return {}
    warm = _get(m, "conditions", "warm", default={}) or {}
    pixel = _get(m, "conditions", "pixel", default={}) or {}
    warm_frac = warm.get("frac_rel_mean", warm.get("frac_rel", m.get("headline_ratio")))
    pixel_frac = pixel.get("frac_rel_mean", pixel.get("frac_rel"))
    return {"warm_frac_rel": warm_frac, "pixel_frac_rel": pixel_frac}


def extract_c9(m):
    if not m:
        return {}
    pred = m.get("predictability", {}) or {}
    model = m.get("primary_model")
    if model not in pred:
        model = "conv" if "conv" in pred else (next(iter(pred), None))
    if model is None or model not in pred:
        return {}
    code = _get(pred, model, "code", default={}) or {}
    pixel = _get(pred, model, "pixel", default={}) or {}
    code_r2 = code.get("r2_vs_copy_last_mean", code.get("r2_vs_copy_last"))
    pixel_r2 = pixel.get("r2_vs_copy_last_mean", pixel.get("r2_vs_copy_last"))
    return {"code_r2": code_r2, "pixel_r2": pixel_r2, "model": model}


EXTRACTORS = {"c1": extract_c1, "c5": extract_c5, "c9": extract_c9}


def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def print_markdown_table(rows, tags):
    cols = ["arm"]
    for tag in tags:
        if tag == "c1":
            cols += ["c1.long_mse", "c1.copy_last_floor", "c1.tf_mse"]
        elif tag == "c5":
            cols += ["c5.warm_frac_rel", "c5.pixel_frac_rel"]
        elif tag == "c9":
            cols += ["c9.code_r2", "c9.pixel_r2"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        vals = [r["arm"]]
        for tag in tags:
            if tag == "c1":
                c1 = r.get("c1", {})
                vals += [_fmt(c1.get("long_mse")), _fmt(c1.get("copy_last_floor")), _fmt(c1.get("tf_mse"))]
            elif tag == "c5":
                c5 = r.get("c5", {})
                vals += [_fmt(c5.get("warm_frac_rel"), 3), _fmt(c5.get("pixel_frac_rel"), 3)]
            elif tag == "c9":
                c9 = r.get("c9", {})
                vals += [_fmt(c9.get("code_r2"), 3), _fmt(c9.get("pixel_r2"), 3)]
        print("| " + " | ".join(vals) + " |")


def main():
    args = parse_args()
    root = lab_root()
    cfg = resolve_config(smoke=bool(args.smoke), root=root)
    arms = arm_list(cfg, args.arms)
    only = args.only or cfg.get("ladder", {}).get("only", DEFAULT_ONLY)

    ran = {tag: False for tag in only}
    if not args.skip_run:
        for arm in arms:
            for tag in only:
                ran[tag] = run_one(tag, arm, args.smoke, root) or ran[tag]
    else:
        ran = {tag: True for tag in only}

    rows = []
    for arm in arms:
        entry = {"arm": arm}
        for tag in only:
            exp_name = EXP_MAP.get(tag)
            if exp_name is None:
                continue
            m = find_latest_run_for_arm(exp_name, arm, root)
            extractor = EXTRACTORS.get(tag)
            entry[tag] = extractor(m) if extractor else (m or {})
        rows.append(entry)

    print("\nLadder summary:")
    print_markdown_table(rows, only)

    run_dir = new_run_dir(root, "ladder")
    summary = "Ladder | " + " | ".join(
        f"{r['arm']}: c1={_fmt(r.get('c1', {}).get('long_mse'))} "
        f"c5={_fmt(r.get('c5', {}).get('warm_frac_rel'), 3)} "
        f"c9={_fmt(r.get('c9', {}).get('code_r2'), 3)}"
        for r in rows
    )
    metrics = {
        "claim": "ladder",
        "arms": arms,
        "only": only,
        "rows": rows,
        "summary": summary,
    }
    finish_run(run_dir, {"ladder": {"arms": arms, "only": only}}, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
