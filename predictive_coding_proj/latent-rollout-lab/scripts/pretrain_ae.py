#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_config
from src.data import load_splits
from src.models.autoencoder import build_ae
from src.pretrain_ae import save_ae, train_ae
from src.utils import get_device, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    args = p.parse_args()
    cfg = resolve_config(args.config, root=ROOT)
    seed_everything(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "auto"))
    data_root = (ROOT / cfg["data"]["root"]).resolve()
    train, val, test, info = load_splits(
        data_root, cfg["data"]["n_train"], cfg["data"]["n_val"], cfg["data"]["n_test"], cfg.get("seed", 0)
    )
    print(info)
    ae = build_ae(cfg).to(device)
    val_mse = train_ae(ae, train, val, cfg, device, fig_path=ROOT / "artifacts" / "ae_recon.png")
    ckpt = ROOT / cfg["ae"]["checkpoint"]
    save_ae(ckpt, ae, cfg, val_mse)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
