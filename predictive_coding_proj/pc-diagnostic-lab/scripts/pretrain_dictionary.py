#!/usr/bin/env python3
"""Pretrain and pin a frozen deconv dictionary (optional; experiments call this too)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import ensure_dictionary, load_data, parse_args, setup
from src.utils import seed_everything


def main():
    args = parse_args("pretrain spatial dictionary")
    root, cfg, device, seeds = setup(args)
    seed = seeds[0]
    seed_everything(seed)
    cfg["seed"] = seed
    train, val, test, info = load_data(cfg, root, seed)
    print(f"device={device}  {info['n_train']} train  hash={info['split_hash']}")
    deconvs, r_init, mse = ensure_dictionary(cfg, train, device, root)
    print(f"dictionary layers={len(deconvs)}  val_mse={mse}  r0={tuple(r_init[0].shape)}")


if __name__ == "__main__":
    main()
