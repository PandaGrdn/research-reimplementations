#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = [
    "configs/strategy/teacher_forcing.yaml",
    "configs/strategy/curriculum_rollout.yaml",
    "configs/strategy/scheduled_sampling.yaml",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    args = p.parse_args()
    extra = []
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]
    if args.n_train is not None:
        extra += ["--n-train", str(args.n_train)]
    extra += ["--seeds", *[str(s) for s in args.seeds]]
    for rel in STRATEGIES:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), str(ROOT / rel), *extra]
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
