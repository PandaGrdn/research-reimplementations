#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_config
from src.data import cache_latents, load_splits, sequence_loader
from src.evaluate import evaluate
from src.models.autoencoder import build_ae
from src.models.registry import build_model, build_strategy
from src.pretrain_ae import load_ae, save_ae, train_ae
from src.train import train_temporal
from src.utils import get_device, git_hash, pip_freeze, seed_everything


def write_yaml(path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def ensure_ae(cfg, train, val, device, root):
    ckpt = root / cfg["ae"]["checkpoint"]
    if ckpt.exists():
        ae, recon = load_ae(ckpt, cfg, device)
        print(f"loaded AE {ckpt} val_recon={recon}")
        return ae, recon
    print("no AE checkpoint; pretraining")
    ae = build_ae(cfg).to(device)
    recon = train_ae(ae, train, val, cfg, device, fig_path=root / "artifacts" / "ae_recon.png")
    save_ae(ckpt, ae, cfg, recon)
    return ae, recon


def run_one(cfg, seed, root):
    cfg = dict(cfg)
    cfg["seed"] = seed
    seed_everything(seed)
    device = get_device(cfg.get("device", "auto"))
    data_root = (root / cfg["data"]["root"]).resolve()
    train, val, test, info = load_splits(
        data_root,
        cfg["data"]["n_train"],
        cfg["data"]["n_val"],
        cfg["data"]["n_test"],
        seed,
    )
    print(f"device={device}  {info['n_train']} train / {info['n_val']} val / {info['n_test']} test  T={info['T']}  hash={info['split_hash']}")

    ae, recon = ensure_ae(cfg, train, val, device, root)
    train_latents = cache_latents(ae, train, device)
    val_latents = cache_latents(ae, val, device)
    test_latents = cache_latents(ae, test, device)
    train_frames = torch.stack(train)
    val_frames = torch.stack(val)
    test_frames = torch.stack(test)

    model = build_model(cfg).to(device)
    strategy = build_strategy(cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.get("run_name") or cfg["strategy"]["name"]
    run_dir = root / "results" / f"{name}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", cfg)
    env = {"git": git_hash(root.parents[1]), "pip": pip_freeze(), "split": info}
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))

    train_temporal(
        model, ae, train_frames, train_latents, val_frames, val_latents,
        strategy, cfg, device, run_dir=run_dir,
    )

    e = cfg["eval"]
    test_loader = sequence_loader(test_frames, test_latents, e.get("batch_size", 8), shuffle=False)
    metrics = evaluate(
        model,
        ae,
        test_loader,
        context_lengths=tuple(e.get("context_lengths", [2, 8])),
        horizons=tuple(e.get("horizons", [1, 5, 10, 18])),
        lpips=bool(e.get("lpips", True)),
        device=device,
    )
    metrics["ae_recon_mse_ckpt"] = recon
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("ae_recon_mse", "headline", "ar_first_vs_tf")}, indent=2))
    print(f"wrote {run_dir}")
    return run_dir, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default=str(ROOT / "configs" / "strategy" / "curriculum_rollout.yaml"))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    cfg = resolve_config(args.config, root=ROOT)
    if args.epochs is not None:
        cfg["temporal"]["epochs"] = args.epochs
    if args.n_train is not None:
        cfg["data"]["n_train"] = args.n_train
    if args.name:
        cfg["run_name"] = args.name
    seeds = args.seeds if args.seeds is not None else [cfg.get("seed", 0)]
    for seed in seeds:
        run_one(cfg, seed, ROOT)


if __name__ == "__main__":
    main()
