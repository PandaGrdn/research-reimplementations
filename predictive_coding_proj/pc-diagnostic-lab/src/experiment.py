"""Shared experiment plumbing: data, dictionary, temporal training, CLI."""

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import lab_root, resolve_config, scheduled_value
from src.data import load_splits, sample_pretrain_frames
from src.rollout import validate_hierarchical, validate_hierarchical_long
from src.spatial_pc import (
    dictionary_key,
    freeze_dictionary,
    load_dictionary,
    make_from_cfg,
    pretrain_spatial,
    save_dictionary,
    unfreeze_dictionary,
)
from src.temporal import CoupledTopDownRNN, TemporalConvRNN, inference_kwargs, train_video_sequence
from src.utils import clone_r, get_device, seed_everything
from src.viz import save_rollout_panel


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--overlay", nargs="+", default=None)
    p.add_argument("--ckpt", default=None, help="load a saved temporal.pt and skip training")
    return p.parse_args()


def setup(args, n_seeds_key="n_seeds"):
    root = lab_root()
    cfg = resolve_config(overlays=getattr(args, "overlay", None), smoke=bool(args.smoke), root=root)
    if args.device:
        cfg["device"] = args.device
    if args.n_train is not None:
        cfg["data"]["n_train"] = args.n_train
    if args.epochs is not None:
        cfg["temporal"]["epochs"] = args.epochs
    device = get_device(cfg.get("device", "auto"))
    n_seeds = cfg.get(n_seeds_key, cfg.get("n_seeds", 3))
    seeds = args.seeds if args.seeds is not None else list(range(n_seeds))
    n_train = cfg["data"]["n_train"]
    n_test = cfg["data"]["n_test"]
    n_rollout = cfg.get("eval", {}).get("n_rollout_sequences", n_test)
    if not args.smoke and (n_train < 1000 or n_test < 100 or n_rollout < 100):
        print(
            f"WARNING: n_train={n_train} n_test={n_test} eval.n_rollout_sequences={n_rollout} is below "
            f"the reporting floor (≥1000 train sequences, ≥100 held-out; rollouts are scored on "
            f"n_rollout_sequences — add --overlay configs/reporting.yaml). Do not quote as a claim."
        )
    return root, cfg, device, seeds


def load_data(cfg, root, seed):
    data_root = (Path(root) / cfg["data"]["root"]).resolve()
    train, val, test, info = load_splits(
        data_root,
        cfg["data"]["n_train"],
        cfg["data"]["n_val"],
        cfg["data"]["n_test"],
        seed,
        data_cfg=cfg["data"],
    )
    return train, val, test, info


def _rng_snapshot():
    snap = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap, seed):
    random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.set_rng_state(snap["torch"])
    if "cuda" in snap:
        torch.cuda.set_rng_state_all(snap["cuda"])
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # The MPS generator cannot be snapshotted. Every script calls
        # seed_everything(seed) right before ensure_dictionary, so reseeding
        # puts it back where a cache hit would have left it.
        torch.mps.manual_seed(seed)


# Fingerprint of the exact dictionary weights each seed used in this process.
# The cache key names a dictionary by its config, but two machines that each
# pretrain one get different weights under the same key. Runs record these so
# a merge, or a model trained on one dictionary, can refuse a mismatch.
DICT_FINGERPRINTS = {}


def dictionary_fingerprint(deconvs):
    h = hashlib.sha256()
    for d in deconvs:
        h.update(d.weight.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def ensure_dictionary(cfg, train, device, root, log=print, seed=None):
    """Load the cached dictionary for this config and seed, pretraining it first
    on a cache miss. A miss and a hit leave the caller in the same RNG state
    and return the dictionary through the same load path, so a run's numbers do
    not depend on whether the cache happened to be warm."""
    seed = cfg["seed"] if seed is None else seed
    s = cfg["spatial"]
    ckpt = Path(root) / s.get("checkpoint", "artifacts/dictionary.pt")
    key = dictionary_key(cfg, seed=seed)
    keyed = ckpt.with_name(f"{ckpt.stem}_{key}{ckpt.suffix}")
    path = keyed if keyed.exists() else ckpt

    if not s.get("pretrain", True):
        seed_everything(seed)  # so r_init / deconv init differ per seed
        _, r_init, deconvs = make_from_cfg(cfg, device, image=train[0][:1])
        if s.get("freeze", True):
            freeze_dictionary(deconvs)
        return deconvs, r_init, None

    if not path.exists():
        rng = _rng_snapshot()
        seed_everything(seed)  # so r_init / deconv init differ per seed
        _, r_init, deconvs = make_from_cfg(cfg, device, image=train[0][:1])
        inf = cfg["inference"]
        frames = sample_pretrain_frames(train, s.get("n_frames", 120), seed=seed)
        frames = [fr.to(device) for fr in frames]
        deconvs, val_mse, _ = pretrain_spatial(
            frames,
            r_init,
            deconvs,
            inf["alpha"],
            inf["lambda_u"],
            inf["lr_r"],
            inf["lr_u"],
            inf["sigma_2"],
            s["epochs"],
            s.get("num_epochs_inner", 50),
            s["num_layers"],
            use_prior=inf.get("use_prior", True),
            log=log,
        )
        # Write-then-rename so a concurrent job never loads a half-written file.
        tmp = keyed.with_name(f"{keyed.name}.{os.getpid()}.tmp")
        save_dictionary(tmp, deconvs, r_init, cfg, val_mse, seed=seed)
        os.replace(tmp, keyed)
        log(f"saved dictionary {keyed}")
        _rng_restore(rng, seed)
        path = keyed

    deconvs, r_init, val_mse = load_dictionary(path, device)
    DICT_FINGERPRINTS[int(seed)] = dictionary_fingerprint(deconvs)
    log(f"loaded dictionary {path} val_mse={val_mse} fingerprint={DICT_FINGERPRINTS[int(seed)]}")
    if s.get("freeze", True):
        freeze_dictionary(deconvs)
    else:
        unfreeze_dictionary(deconvs)
    return deconvs, r_init, val_mse


def arm_name(cfg):
    return cfg.get("arm", "baseline")


def build_temporal(cfg, r_init, device):
    t = cfg["temporal"]
    return TemporalConvRNN(
        r_init,
        delta_scale=t.get("delta_scale", 1.0),
        delta_bounded=t.get("delta_bounded", True),
    ).to(device)


def save_temporal_checkpoint(path, temporal_nn, cfg, seed=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_kind = "coupled" if isinstance(temporal_nn, CoupledTopDownRNN) else "independent"
    torch.save(
        {
            "temporal_state": temporal_nn.state_dict(),
            "temporal": cfg.get("temporal"),
            "inference": cfg.get("inference"),
            "spatial": cfg.get("spatial"),
            "seed": seed,
            "dictionary_key": dictionary_key(cfg, seed=seed),
            "dictionary_fingerprint": DICT_FINGERPRINTS.get(int(seed)),
            "model_kind": model_kind,
        },
        path,
    )
    return path


def load_temporal_checkpoint(path, r_init, cfg, device, deconvs=None):
    """Loads either checkpoint kind, dispatching on the stored `model_kind`
    ("independent" or "coupled"; missing key -> "independent", so pre-rework
    checkpoints like artifacts/offline_gru_seed0.pt still load unchanged).
    "coupled" checkpoints need `deconvs` (the dictionary) to reconstruct
    `CoupledTopDownRNN`'s top-down generation — a missing `deconvs` there is
    a caller bug, so it raises rather than silently building the wrong model.
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model_kind = ckpt.get("model_kind", "independent")
    saved_fp = ckpt.get("dictionary_fingerprint")
    current_fp = DICT_FINGERPRINTS.get(int(ckpt.get("seed", -1)))
    if saved_fp and current_fp and saved_fp != current_fp:
        raise SystemExit(
            f"{path} was trained with dictionary {saved_fp}, but this run loaded dictionary {current_fp} "
            f"for seed {ckpt.get('seed')}. Same cache key, different weights: run on the machine whose "
            f"artifacts/ trained this model."
        )
    if model_kind == "coupled":
        if deconvs is None:
            raise ValueError(
                f"checkpoint {path} is a coupled model (model_kind='coupled') but "
                "load_temporal_checkpoint was not given `deconvs` — CoupledTopDownRNN "
                "needs the dictionary to reconstruct its top-down lower-layer predictions."
            )
        t = cfg.get("temporal") or {}
        model = CoupledTopDownRNN(
            r_init,
            deconvs,
            delta_scale=t.get("delta_scale", 1.0),
            delta_bounded=t.get("delta_bounded", True),
        ).to(device)
    else:
        model = build_temporal(cfg, r_init, device)
    model.load_state_dict(ckpt["temporal_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-epoch training checkpoints (resume after a time limit or a lost node)
# ---------------------------------------------------------------------------
# Active only when PC_LAB_RUN_DIR is set, i.e. under scripts/run_parallel.py or
# Slurm, where a rerun of the same job lands in the same directory. The k-th
# train_temporal_pc call in a process uses train_ckpt/call<k>.pt; scripts call
# it in a fixed order, so a rerun maps each call to its own checkpoint. On CPU
# and CUDA a resumed run is bit-identical to an uninterrupted one (tested). The
# MPS generator cannot be saved, so on MPS a resume is close but not exact.
_TRAIN_CALLS = [0]


def _train_ckpt_path():
    base = os.environ.get("PC_LAB_RUN_DIR")
    if not base or os.environ.get("PC_LAB_NO_TRAIN_CKPT") == "1":
        return None
    _TRAIN_CALLS[0] += 1
    return Path(base) / "train_ckpt" / f"call{_TRAIN_CALLS[0]}.pt"


def _train_fingerprint(cfg, sequences, val_seq):
    blob = json.dumps(
        {
            "sections": {k: cfg.get(k) for k in ("data", "spatial", "inference", "temporal")},
            "seed": cfg.get("seed"),
            "n_sequences": len(sequences),
            "has_val": val_seq is not None,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _save_train_ckpt(path, epoch_done, n_epochs, fingerprint, temporal_nn, deconvs, encoder, history, last_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch_done": epoch_done,
        "n_epochs": n_epochs,
        "fingerprint": fingerprint,
        "temporal_state": temporal_nn.state_dict(),
        "deconv_state": [d.state_dict() for d in deconvs],
        "encoder_state": encoder.state_dict() if encoder is not None and hasattr(encoder, "state_dict") else None,
        "history": history,
        "last_val": last_val,
        "rng": _rng_snapshot(),
    }
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_train_ckpt(path, fingerprint, temporal_nn, deconvs, encoder, device, log):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state["fingerprint"] != fingerprint:
        raise SystemExit(
            f"{path} was written by a run with a different config or data; refusing to resume from it. "
            f"Delete it or use a fresh run directory."
        )
    temporal_nn.load_state_dict(state["temporal_state"])
    for d, sd in zip(deconvs, state["deconv_state"]):
        d.load_state_dict(sd)
    if encoder is not None and state.get("encoder_state") is not None:
        encoder.load_state_dict(state["encoder_state"])
    snap = state["rng"]
    random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.set_rng_state(snap["torch"].cpu() if hasattr(snap["torch"], "cpu") else snap["torch"])
    if "cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in snap["cuda"]])
    if str(device).startswith("mps"):
        log("note: resuming on MPS; its random generator is not saved, so this run is not bit-identical")
    log(f"resumed training from {path} at epoch {state['epoch_done']}/{state['n_epochs']}")
    return state["epoch_done"], state["history"], state["last_val"]


def train_temporal_pc(
    sequences,
    r_init,
    deconvs,
    temporal_nn,
    cfg,
    device,
    val_seq=None,
    log=print,
    run_dir=None,
    encoder=None,
):
    """Notebook-style epoch loop. Returns per-epoch motion_gap / delta_ratio history."""
    tcfg = cfg["temporal"]
    inf = cfg["inference"]
    n_epochs = tcfg["epochs"]
    kw = inference_kwargs(cfg)
    history = []
    last_val = None

    start_epoch = 0
    ckpt_path = _train_ckpt_path()
    fingerprint = _train_fingerprint(cfg, sequences, val_seq) if ckpt_path is not None else None
    if ckpt_path is not None and ckpt_path.exists():
        start_epoch, history, last_val = _load_train_ckpt(
            ckpt_path, fingerprint, temporal_nn, deconvs, encoder, device, log
        )

    for epoch in range(start_epoch, n_epochs):
        ss_p = scheduled_value(tcfg.get("ss_p", 0.0), tcfg.get("ss_p_end"), epoch, n_epochs)
        r_noise = scheduled_value(tcfg.get("r_noise_std", 0.0), tcfg.get("r_noise_std_end"), epoch, n_epochs)
        epoch_ratios = []
        for seq in sequences:
            r_curr = clone_r(r_init)
            seq_dev = seq.to(device)
            _, train_stats = train_video_sequence(
                seq_dev,
                r_curr,
                kw["alpha"],
                kw["lambda_u"],
                kw["lr_r"],
                kw["lr_u"],
                kw["sigma_2"],
                kw["num_epochs_outer"],
                kw["num_epochs_inner"],
                kw["num_layers"],
                deconvs,
                temporal_nn,
                r_noise_std=r_noise,
                ss_p=ss_p,
                rollout_k=kw["rollout_k"],
                lambda_slow=kw["lambda_slow"],
                delta_loss_weight=kw["delta_loss_weight"],
                delta_target_loss=kw["delta_target_loss"],
                use_prior=kw["use_prior"],
                max_grad_norm=kw["max_grad_norm"],
                temporal_prior_weight=kw["temporal_prior_weight"],
                slow_unroll_k=kw["slow_unroll_k"],
                encoder=encoder,
            )
            epoch_ratios.append(train_stats["ratio"])
        train_ratio = sum(epoch_ratios) / max(len(epoch_ratios), 1)
        rec = {
            "epoch": epoch,
            "ss_p": ss_p,
            "r_noise_std": r_noise,
            "train_delta_ratio": train_ratio,
        }
        if val_seq is not None:
            last_val = validate_hierarchical(
                val_seq.to(device),
                r_init,
                inf["num_epochs_inner"],
                cfg["spatial"]["num_layers"],
                inf["sigma_2"],
                inf["alpha"],
                inf["lr_r"],
                deconvs,
                temporal_nn,
                use_prior=inf.get("use_prior", True),
                temporal_prior_weight=inf.get("temporal_prior_weight", 0.01),
                encoder=encoder,
                log=log,
            )
            rec["val_mse"] = last_val["mse"]
            rec["motion_gap"] = last_val["motion_gap"]
            rec["delta_ratio"] = last_val["delta_ratio"]
            rec["mse_per_frame"] = last_val["mse_per_frame"]
            rec["copy_last_mse"] = last_val["copy_last_mse"]
            rec["mean_frame_mse"] = last_val["mean_frame_mse"]
            log(
                f"[Epoch {epoch:03d}/{n_epochs:03d}]  r_noise={r_noise:.4f}  ss_p={ss_p:.3f}  "
                f"Next-frame MSE: {last_val['mse']:.6f}  train_δ-ratio={train_ratio:.4f}  "
                f"val_δ-ratio={last_val['delta_ratio']:.4f}  gap={last_val['motion_gap']:+.6f}"
            )
        else:
            log(f"[Epoch {epoch:03d}/{n_epochs:03d}]  r_noise={r_noise:.4f}  ss_p={ss_p:.3f}  train_δ-ratio={train_ratio:.4f}")
        history.append(rec)
        if ckpt_path is not None:
            _save_train_ckpt(ckpt_path, epoch + 1, n_epochs, fingerprint, temporal_nn, deconvs, encoder, history, last_val)
            stop_after = os.environ.get("PC_LAB_TEST_STOP_AFTER_EPOCH")  # tests simulate a killed job
            if stop_after is not None and epoch + 1 == int(stop_after):
                raise SystemExit(75)

    if run_dir is not None and last_val is not None:
        save_rollout_panel(
            last_val["true_frames"],
            last_val["pred_frames"],
            context=1,
            path=Path(run_dir) / "tf_panel.png",
            n_show=min(8, last_val["true_frames"].shape[0]),
        )
    if run_dir is not None:
        save_temporal_checkpoint(
            Path(run_dir) / "temporal.pt", temporal_nn, cfg, seed=cfg.get("seed", 0)
        )
    return temporal_nn, history, last_val


def eval_long_rollouts(
    seqs, r_init, deconvs, temporal_nn, cfg, device, split_points=None, log=print, run_dir=None, encoder=None
):
    inf = cfg["inference"]
    tcfg = cfg["temporal"]
    split_points = split_points or cfg["eval"]["split_points"]
    n_eval = min(len(seqs), cfg["eval"].get("n_rollout_sequences", len(seqs)))
    temporal_prior_weight = inf.get("temporal_prior_weight", 0.01)
    out = {}
    for sp in split_points:
        curves, long_mses, saturations = [], [], []
        copy_last_long_vals, mean_frame_long_vals = [], []
        copy_last_curves, mean_frame_curves = [], []
        blank_long_vals, blank_curves, mass_ratio_long_vals, mass_ratio_curves = [], [], [], []
        corr_long_vals, corr_curves, cl_corr_long_vals, cl_corr_curves = [], [], [], []
        last = None
        for seq in seqs[:n_eval]:
            last = validate_hierarchical_long(
                seq.to(device),
                r_init,
                inf["num_epochs_inner"],
                cfg["spatial"]["num_layers"],
                inf["sigma_2"],
                inf["alpha"],
                inf["lr_r"],
                deconvs,
                temporal_nn,
                split_point=sp,
                split_fix=tcfg.get("split_fix", False),
                use_prior=inf.get("use_prior", True),
                temporal_prior_weight=temporal_prior_weight,
                encoder=encoder,
                log=log,
            )
            curves.append(last["mse_per_frame"])
            long_mses.append(last["long_mse"])
            copy_last_long_vals.append(last["copy_last_long_mse"])
            mean_frame_long_vals.append(last["mean_frame_long_mse"])
            copy_last_curves.append(last["copy_last_mse_per_frame"])
            mean_frame_curves.append(last["mean_frame_mse_per_frame"])
            blank_long_vals.append(last["blank_long_mse"])
            blank_curves.append(last["blank_mse_per_frame"])
            mass_ratio_long_vals.append(last["mass_ratio_long"])
            mass_ratio_curves.append(last["mass_ratio_per_frame"])
            corr_long_vals.append(last["corr_long"])
            corr_curves.append(last["corr_per_frame"])
            cl_corr_long_vals.append(last["copy_last_corr_long"])
            cl_corr_curves.append(last["copy_last_corr_per_frame"])
            if last["saturation"] is not None:
                saturations.append(last["saturation"])
        from src.metrics import stack_mean_std
        from src.utils import mean_std

        mu, sd = stack_mean_std(curves)
        long_mu, long_sd = mean_std(long_mses)
        cl_long_mu, cl_long_sd = mean_std(copy_last_long_vals)
        mf_long_mu, mf_long_sd = mean_std(mean_frame_long_vals)
        cl_curve_mu, _ = stack_mean_std(copy_last_curves)
        mf_curve_mu, _ = stack_mean_std(mean_frame_curves)
        out[str(sp)] = {
            "mse_per_frame_mean": mu,
            "mse_per_frame_std": sd,
            "long_mse_mean": long_mu,
            "long_mse_std": long_sd,
            "copy_last_long_mse_mean": cl_long_mu,
            "copy_last_long_mse_std": cl_long_sd,
            "mean_frame_long_mse_mean": mf_long_mu,
            "mean_frame_long_mse_std": mf_long_sd,
            "copy_last_mse_per_frame_mean": cl_curve_mu,
            "mean_frame_mse_per_frame_mean": mf_curve_mu,
            "blank_long_mse_mean": mean_std(blank_long_vals)[0],
            "blank_long_mse_std": mean_std(blank_long_vals)[1],
            "blank_mse_per_frame_mean": stack_mean_std(blank_curves)[0],
            "mass_ratio_long_mean": mean_std(mass_ratio_long_vals)[0],
            "mass_ratio_long_std": mean_std(mass_ratio_long_vals)[1],
            "mass_ratio_per_frame_mean": stack_mean_std(mass_ratio_curves)[0],
            "corr_long_mean": mean_std(corr_long_vals)[0],
            "corr_long_std": mean_std(corr_long_vals)[1],
            "corr_per_frame_mean": stack_mean_std(corr_curves)[0],
            "copy_last_corr_long_mean": mean_std(cl_corr_long_vals)[0],
            "copy_last_corr_per_frame_mean": stack_mean_std(cl_corr_curves)[0],
            "n": n_eval,
            "saturation": saturations[0] if saturations else None,
        }
        if run_dir is not None and last is not None:
            save_rollout_panel(
                last["true_frames"],
                last["pred_frames"],
                context=sp,
                path=Path(run_dir) / f"long_split{sp}.png",
                n_show=min(12, last["true_frames"].shape[0]),
            )
    return out


def overlay_temporal(cfg, **kwargs):
    out = copy.deepcopy(cfg)
    out["temporal"].update(kwargs)
    return out
