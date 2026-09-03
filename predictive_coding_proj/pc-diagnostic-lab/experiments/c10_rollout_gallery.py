#!/usr/bin/env python3
"""C10 — look at the pixels.

Saves the trained temporal net, runs teacher-forced and closed-loop rollout on
held-out sequences, and writes inspection grids (ground truth / TF / AR /
copy-last) so a human can see whether the numbers are hiding a blur.

Reuse a checkpoint instead of retraining:

    python experiments/c10_rollout_gallery.py --ckpt results/c10_rollout_gallery/latest/temporal.pt --n-train 100 --device mps
"""

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    load_temporal_checkpoint,
    parse_args,
    save_temporal_checkpoint,
    setup,
    train_temporal_pc,
)
from src.rollout import validate_hierarchical, validate_hierarchical_long
from src.utils import finish_run, new_run_dir, seed_everything
from src.viz import apply_pub_style, save_rollout_inspection


def copy_last_frames(seq, split_point):
    """Per-column copy-last baseline matching true_frames[k] <-> seq[k+1]."""
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]
    freeze_idx = max(split_point - 1, 0)
    frames = []
    for t in range(1, T):
        src = min(t - 1, freeze_idx)
        frames.append((seq[src] - seq[src].mean()).detach().clone())
    return torch.stack(frames, dim=0)


def rollout_kwargs(cfg):
    inf = cfg["inference"]
    return dict(
        num_epochs_inner=inf["num_epochs_inner"],
        num_layers=cfg["spatial"]["num_layers"],
        sigma_2=inf["sigma_2"],
        alpha=inf["alpha"],
        lr_r=inf["lr_r"],
        use_prior=inf.get("use_prior", True),
        temporal_prior_weight=inf.get("temporal_prior_weight", 0.01),
    )


def main():
    args = parse_args("C10 visual autoregressive rollout gallery")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c10_rollout_gallery", arm=arm_name(cfg))
    gallery_dir = run_dir / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    seed = seeds[0]
    seed_everything(seed)
    cfg["seed"] = seed
    train, val, test, info = load_data(cfg, root, seed)
    print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
    deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

    ckpt_in = args.ckpt
    if ckpt_in:
        print(f"loading temporal checkpoint {ckpt_in}")
        model = load_temporal_checkpoint(ckpt_in, r_init, cfg, device)
        trained = False
    else:
        model = build_temporal(cfg, r_init, device)
        model, history, _ = train_temporal_pc(
            train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print, run_dir=run_dir
        )
        trained = True

    ckpt_out = run_dir / "temporal.pt"
    if not ckpt_out.exists():
        save_temporal_checkpoint(ckpt_out, model, cfg, seed=seed)
    artifacts = Path(root) / "artifacts" / f"temporal_seed{seed}.pt"
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_out, artifacts)
    print(f"saved {ckpt_out}  (also {artifacts})")

    headline_sp = int(cfg["eval"].get("headline_split", 10))
    n_show = min(len(test), cfg.get("c10", {}).get("n_sequences", 4))
    kw = rollout_kwargs(cfg)
    split_fix = cfg["temporal"].get("split_fix", False)
    gallery_paths = []
    per_seq = []

    for i, seq in enumerate(test[:n_show]):
        seq_dev = seq.to(device)
        tf = validate_hierarchical(seq_dev, r_init, deconvs=deconvs, temporal_nn=model, log=lambda *a, **k: None, **kw)
        ar = validate_hierarchical_long(
            seq_dev, r_init, deconvs=deconvs, temporal_nn=model,
            split_point=headline_sp, split_fix=split_fix, log=lambda *a, **k: None, **kw,
        )
        cl = copy_last_frames(seq_dev, headline_sp)
        path = gallery_dir / f"seq{i:02d}_split{headline_sp}.png"
        save_rollout_inspection(
            ar["true_frames"], tf["pred_frames"], ar["pred_frames"], cl,
            split_point=headline_sp, path=path,
        )
        gallery_paths.append(str(path.relative_to(root)))
        rec = {
            "index": i,
            "tf_mse": tf["mse"],
            "long_mse": ar["long_mse"],
            "copy_last_long_mse": ar["copy_last_long_mse"],
            "mean_frame_long_mse": ar["mean_frame_long_mse"],
            "path": str(path.relative_to(root)),
        }
        per_seq.append(rec)
        print(
            f"C10 | seq {i}: tf={rec['tf_mse']:.4f}  AR long={rec['long_mse']:.4f}  "
            f"copy-last={rec['copy_last_long_mse']:.4f}  -> {path.name}"
        )

    headline = gallery_dir / f"seq00_split{headline_sp}.png"
    fig_dst = Path(root) / "figures" / "fig_c10.png"
    if headline.exists():
        fig_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(headline, fig_dst)

    summary = (
        f"C10 | n={n_show} split={headline_sp}  "
        f"AR long MSE {sum(s['long_mse'] for s in per_seq) / max(n_show, 1):.4f}  "
        f"copy-last {sum(s['copy_last_long_mse'] for s in per_seq) / max(n_show, 1):.4f}  "
        f"| grids in {gallery_dir}  ckpt={ckpt_out}  trained={trained}"
    )
    metrics = {
        "claim": "C10",
        "headline_split": headline_sp,
        "checkpoint": str(ckpt_out.relative_to(root)),
        "loaded_ckpt": ckpt_in,
        "trained": trained,
        "gallery": per_seq,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
