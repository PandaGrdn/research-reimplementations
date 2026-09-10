#!/usr/bin/env python3
"""C12 — per-term PC energy under closed-loop fade, with an artificial-fade
control so "energy rises as it fades" can be told apart from "energy is just
monotone in contrast" (see SPEC2 background: at full precision, a real digit
scaled by 0.5/0.25/0 has energy 0.065/0.016/0 vs 0.26 at full contrast, and
the ~250 spike previously reported under rollout was a DC-centering artifact,
not a reconstruction failure).

Three things happen per seed:
  1. contrast_sweep — settle from zero init directly on scale*I for held-out
     frames, scale in c12.fade_scales. No rollout involved at all; this is
     the control every rollout energy curve below has to be read against.
  2. energy_fade_rollout on the "independent" model (artifacts/offline_gru_seed{k}.pt,
     the 2-layer TemporalConvRNN with independent per-layer cells — Agent A's
     C11 checkpoint), falling back to training one in-process if absent.
  3. The same rollout on the "coupled" model (artifacts/offline_gru_coupled_seed{k}.pt,
     Agent A's CoupledTopDownRNN, top-down generation only) if that checkpoint
     exists — its inter-layer term e1 is exactly zero by construction, which
     is the reference point for how much of the independent model's e1 growth
     is a real coupling failure vs settle noise.

    python experiments/c12_energy_fade.py --smoke
    python experiments/c12_energy_fade.py --device mps
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.energy_fade import (
    _check_energy_terms,
    aggregate_contrast_sweep,
    aggregate_energy_fade,
    contrast_sweep,
    energy_fade_rollout,
)
from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    load_temporal_checkpoint,
    parse_args,
    save_temporal_checkpoint,
    setup,
)
from src.inference import collect_eval_frames
from src.offline_gru import encode_eval_codes, train_offline_gru
from src.utils import finish_run, new_run_dir, seed_everything

# Agent A is concurrently adding cache_eval_codes to src/offline_gru.py; use it
# once it lands (shared cached codes across C11/C12/C13), otherwise fall back
# to the uncached encoder so this script runs either way.
try:
    from src.offline_gru import cache_eval_codes
except ImportError:
    cache_eval_codes = None

_VERDICT_KEYS = (
    "e1_ratio_post",
    "e1_ratio_pre",
    "dc_artifact_share_post",
    "fade_then_diverge",
    "energy_norm_tracks_fade",
    "e_norm_pred_pre",
    "e_norm_pred_post",
)


def _get_train_test_codes(train_seqs, test_seqs, deconvs, r_init, cfg, root, seed):
    if cache_eval_codes is not None:
        try:
            train_codes = cache_eval_codes("train", train_seqs, deconvs, r_init, cfg, root, seed)
            test_codes = cache_eval_codes("test", test_seqs, deconvs, r_init, cfg, root, seed)
            return train_codes, test_codes
        except Exception as e:  # defensive: cache_eval_codes may still be mid-implementation
            print(f"cache_eval_codes failed ({type(e).__name__}: {e}); falling back to encode_eval_codes")
    return (
        encode_eval_codes(train_seqs, deconvs, r_init, cfg),
        encode_eval_codes(test_seqs, deconvs, r_init, cfg),
    )


def _load_checkpoint(path, r_init, cfg, device, deconvs=None):
    """Agent A is concurrently adding a `deconvs=` kwarg to load_temporal_checkpoint
    (needed to construct CoupledTopDownRNN). Try the new signature first, fall
    back to the old one so this script works whether or not that change has
    landed yet."""
    try:
        return load_temporal_checkpoint(path, r_init, cfg, device, deconvs=deconvs)
    except TypeError:
        return load_temporal_checkpoint(path, r_init, cfg, device)


def _run_rollouts(seqs, r_init, deconvs, model, cfg, split_point, seed):
    per_seq = []
    for i, seq in enumerate(seqs):
        rec = energy_fade_rollout(seq, r_init, deconvs, model, cfg, split_point=split_point)
        rec["index"] = i
        rec["seed"] = seed
        per_seq.append(rec)
    return per_seq


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "n/a"


def main():
    args = parse_args("C12 per-term energy vs closed-loop fade, with artificial-fade control")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c12_energy_fade", arm=arm_name(cfg))

    _check_energy_terms()

    c11 = cfg.get("c11", {})
    c12 = cfg.get("c12", {})
    headline_sp = int(cfg["eval"].get("headline_split", 10))
    n_eval = c12.get("n_sequences", 32)
    n_control_frames = c12.get("n_control_frames", 20)
    fade_scales = c12.get("fade_scales", [1.0, 0.75, 0.5, 0.25, 0.1, 0.0])
    epochs = args.epochs if args.epochs is not None else c11.get("epochs", cfg["temporal"]["epochs"])

    term_keys = ("e0", "e1", "prior", "total", "dc_offset", "e0_dc_free")
    control_pool = {s: {k: [] for k in term_keys + ("r_norm",)} for s in fade_scales}
    control_per_seed = []

    indep_per_seq_all, coupled_per_seq_all = [], []
    indep_per_seed, coupled_per_seed = [], []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        n_use = min(n_eval, len(test))
        eval_seqs = [s.to(device) for s in test[:n_use]]

        # -- 1. artificial-fade control (no rollout) --
        control_frames = collect_eval_frames(eval_seqs, n_control_frames)
        seed_control = contrast_sweep(control_frames, r_init, deconvs, cfg, fade_scales)
        for s in fade_scales:
            for k in control_pool[s]:
                control_pool[s][k].extend(seed_control[s][k])
        seed_control_agg = aggregate_contrast_sweep(seed_control, fade_scales)
        control_per_seed.append({"seed": seed, **seed_control_agg})
        print(
            f"  seed={seed} control: blank_is_minimum={seed_control_agg['blank_is_minimum']}  "
            f"energy@scale={seed_control_agg['energy_at_scale']}"
        )

        # -- 2. independent model (Agent A's C11 checkpoint, or a fallback trained here) --
        ckpt = Path(root) / "artifacts" / f"offline_gru_seed{seed}.pt"
        trained = False
        if ckpt.exists():
            print(f"  seed={seed} loading independent checkpoint {ckpt}")
            model = _load_checkpoint(ckpt, r_init, cfg, device, deconvs=deconvs)
        else:
            print(f"  seed={seed} no independent checkpoint at {ckpt}; training a fallback offline GRU")
            n_train_seq = c11.get("n_train_sequences", cfg["data"]["n_train"])
            n_test_seq = min(len(test), c11.get("n_test_sequences", cfg["data"]["n_test"]))
            train_seqs_full = [s.to(device) for s in train[:n_train_seq]]
            test_seqs_full = [s.to(device) for s in test[:n_test_seq]]
            train_codes, test_codes = _get_train_test_codes(
                train_seqs_full, test_seqs_full, deconvs, r_init, cfg, root, seed
            )
            model = build_temporal(cfg, r_init, device)
            train_offline_gru(
                model,
                train_codes,
                test_codes,
                epochs=epochs,
                lr=c11.get("lr", 1e-3),
                device=device,
                target_r2=c11.get("target_r2", 0.5),
                log=print,
            )
            trained = True
            save_temporal_checkpoint(run_dir / f"temporal_seed{seed}.pt", model, cfg, seed=seed)

        per_seq_indep = _run_rollouts(eval_seqs, r_init, deconvs, model, cfg, headline_sp, seed)
        indep_per_seq_all.extend(per_seq_indep)
        seed_agg_indep = aggregate_energy_fade(per_seq_indep, headline_sp)
        indep_per_seed.append(
            {
                "seed": seed,
                "trained": trained,
                "checkpoint": str(ckpt) if ckpt.exists() else None,
                "n_sequences": n_use,
                **{k: seed_agg_indep[k] for k in _VERDICT_KEYS},
            }
        )
        print(
            f"  seed={seed} independent: e1 pred/true post={_fmt(seed_agg_indep['e1_ratio_post'], 3)}  "
            f"dc_artifact_share_post={_fmt(seed_agg_indep['dc_artifact_share_post'], 3)}  "
            f"fade_then_diverge={seed_agg_indep['fade_then_diverge']}"
        )

        # -- 3. coupled model (Agent A's CoupledTopDownRNN), if a checkpoint exists --
        coupled_ckpt = Path(root) / "artifacts" / f"offline_gru_coupled_seed{seed}.pt"
        if coupled_ckpt.exists():
            coupled_model = None
            try:
                coupled_model = _load_checkpoint(coupled_ckpt, r_init, cfg, device, deconvs=deconvs)
            except Exception as e:
                print(f"  seed={seed}: could not load coupled checkpoint ({type(e).__name__}: {e}); skipping")
            if coupled_model is not None:
                per_seq_coupled = _run_rollouts(eval_seqs, r_init, deconvs, coupled_model, cfg, headline_sp, seed)
                coupled_per_seq_all.extend(per_seq_coupled)
                seed_agg_coupled = aggregate_energy_fade(per_seq_coupled, headline_sp)
                coupled_per_seed.append(
                    {
                        "seed": seed,
                        "checkpoint": str(coupled_ckpt),
                        "n_sequences": n_use,
                        **{k: seed_agg_coupled[k] for k in _VERDICT_KEYS},
                    }
                )
                print(
                    f"  seed={seed} coupled: e_norm_pred pre->post="
                    f"{_fmt(seed_agg_coupled['e_norm_pred_pre'])}->{_fmt(seed_agg_coupled['e_norm_pred_post'])}"
                )
        else:
            print(f"  seed={seed}: no coupled checkpoint at {coupled_ckpt}; skipping coupled model")

    control_agg = aggregate_contrast_sweep(control_pool, fade_scales)
    control_agg["per_seed"] = control_per_seed

    models_out = {}
    indep_agg = aggregate_energy_fade(indep_per_seq_all, headline_sp)
    indep_agg["per_seed"] = indep_per_seed
    models_out["independent"] = indep_agg

    if coupled_per_seq_all:
        coupled_agg = aggregate_energy_fade(coupled_per_seq_all, headline_sp)
        coupled_agg["per_seed"] = coupled_per_seed
        models_out["coupled"] = coupled_agg
    else:
        models_out["coupled"] = {
            "skipped": True,
            "reason": "no artifacts/offline_gru_coupled_seed{k}.pt found for any seed",
        }

    e_at = control_agg["energy_at_scale"]
    coupled_line = "coupled: skipped (no checkpoints)"
    if not models_out["coupled"].get("skipped"):
        cg = models_out["coupled"]
        coupled_line = (
            f"coupled: e1=0 by construction, e_norm pre->post="
            f"{_fmt(cg['e_norm_pred_pre'])}->{_fmt(cg['e_norm_pred_post'])}"
        )

    summary = (
        f"C12 | control: energy at scale 1/0.5/0 = "
        f"{_fmt(e_at.get('1.0'))}/{_fmt(e_at.get('0.5'))}/{_fmt(e_at.get('0.0'))} "
        f"(blank_is_minimum={control_agg['blank_is_minimum']}) | "
        f"indep post-split: e1 pred/true={_fmt(indep_agg['e1_ratio_post'])}, "
        f"dc_artifact_share={_fmt(indep_agg['dc_artifact_share_post'])}, "
        f"fade_then_diverge={indep_agg['fade_then_diverge']}, "
        f"e_norm pre->post={_fmt(indep_agg['e_norm_pred_pre'])}->{_fmt(indep_agg['e_norm_pred_post'])} | "
        f"{coupled_line} | seeds={len(seeds)}"
    )

    metrics = {
        "claim": "C12",
        "headline_split": headline_sp,
        "fade_scales": fade_scales,
        "n_control_frames": n_control_frames,
        "n_sequences": n_eval,
        "control": control_agg,
        "models": models_out,
        "seeds": seeds,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
