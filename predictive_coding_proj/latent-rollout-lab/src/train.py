from pathlib import Path

import torch

from src.data import sequence_loader
from src.evaluate import evaluate
from src.viz import save_rollout_panel


def train_temporal(model, ae, train_frames, train_latents, val_frames, val_latents, strategy, cfg, device, log=print, run_dir=None):
    tcfg = cfg["temporal"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loader = sequence_loader(
        train_frames, train_latents, tcfg["batch_size"], shuffle=True, seed=cfg.get("seed", 0)
    )
    val_loader = sequence_loader(
        val_frames, val_latents, cfg["eval"].get("batch_size", tcfg["batch_size"]), shuffle=False
    )
    context = tcfg["context"]
    best = float("inf")
    best_path = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(tcfg["epochs"]):
        model.gru.train()
        model.head.train()
        totals = {}
        n = 0
        last_n_ar = 0
        for frames, latents in loader:
            frames = frames.to(device)
            latents = latents.to(device)
            out = strategy.compute_loss(model, ae, latents, frames, epoch)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            b = frames.shape[0]
            n += b
            last_n_ar = out["logs"].get("n_ar", 0)
            for k, v in out["logs"].items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0.0) + float(v) * b
        strategy.on_epoch_end(epoch)
        logs = {k: totals[k] / max(n, 1) for k in totals if k != "n_ar"}
        tf = logs.get("tf", 0.0)
        pix = logs.get("tf_pixel", 0.0)
        if last_n_ar == 0:
            ar_msg = "TF-only"
        else:
            ar_msg = f"rollout MSE ({last_n_ar} steps): {logs.get('ar', logs.get('ss', 0.0)):.6f}"
        log(f"Epoch {epoch + 1}  teacher-force MSE: {tf:.6f}  pixel: {pix:.6f}  {ar_msg}")

        model.eval()
        val_metrics = evaluate(
            model,
            ae,
            val_loader,
            context_lengths=(context,),
            horizons=tuple(cfg["eval"].get("horizons", [1, 5, 10, 18])),
            lpips=False,
            device=device,
        )
        val_roll = 0.0
        curve = val_metrics.get("ar", {}).get(str(context), {}).get("mse", [])
        if curve:
            val_roll = sum(curve) / len(curve)
        log(f"  val rollout MSE: {val_roll:.6f}")
        if val_roll < best and run_dir is not None:
            best = val_roll
            best_path = run_dir / "best.pt"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_rollout_mse": val_roll}, best_path)

    if best_path is not None and best_path.exists():
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        log(f"loaded best checkpoint epoch {ckpt['epoch'] + 1} val rollout {ckpt['val_rollout_mse']:.6f}")

    if run_dir is not None:
        model.eval()
        frames = val_frames[:1].to(device)
        latents = val_latents[:1].to(device)
        r_tf = model.predict_latent(latents, teacher_force=True)
        r_ar = model.rollout(latents[:, :context], n_steps=latents.shape[1] - context)
        save_rollout_panel(frames[0], ae.decode(r_tf)[0], 2, run_dir / "tf_panel.png")
        save_rollout_panel(frames[0], ae.decode(r_ar)[0], context, run_dir / "ar_panel.png")
    return model
