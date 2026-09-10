#!/usr/bin/env python3
"""Full diagnostic pipeline: dictionary → C1–C8 → figures."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "experiments/c0_isolation_test.py",
    "experiments/c1_rollout_collapse.py",
    "experiments/c2_mitigation_grid.py",
    "experiments/c3_copy_detection.py",
    "experiments/c4_settle_determinism.py",
    "experiments/c5_latent_smoothness.py",
    "experiments/c6_noise_floor.py",
    "experiments/c7_slowness_sweep.py",
    "experiments/c8_amortized_contrast.py",
    "experiments/c9_predictability.py",
    "experiments/c10_rollout_gallery.py",
    "experiments/c11_offline_gru.py",
    "experiments/c12_energy_fade.py",
    "experiments/c13_consistency_rollout.py",
]


def run(cmd, extra):
    full = [sys.executable, str(ROOT / cmd)] + extra
    print("\n>>>", " ".join(full), flush=True)
    subprocess.check_call(full, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny CPU overlay, <10 min")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="subset of c1..c8")
    args = p.parse_args()
    extra = ["--smoke"] if args.smoke else []
    wanted = set(args.only) if args.only else None
    for exp in EXPERIMENTS:
        tag = Path(exp).stem.split("_")[0]
        if wanted and tag not in wanted and Path(exp).stem not in wanted:
            continue
        run(exp, extra)
    if not args.skip_figures:
        run("scripts/make_figures.py", [])


if __name__ == "__main__":
    main()
