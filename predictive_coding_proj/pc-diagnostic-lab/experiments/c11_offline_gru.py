#!/usr/bin/env python3
"""C11 — train two offline ConvGRUs on cached eval-protocol codes: an
"independent" model (`TemporalConvRNN`, one GRU cell + readout per layer) and
a "coupled" model (`CoupledTopDownRNN`, one GRU cell on the top layer only;
the bottom layer is generated top-down through the frozen-for-gradient
dictionary, so its inter-layer consistency error e1 is exactly zero by
construction).

Two things the previous version got wrong (see SPEC2 background):
  - It only ran seeds[0] and early-stopped after one epoch. This version
    runs every seed in `seeds`, trains both models for the FULL `c11.epochs`
    with no early stop, and checkpoints the best-epoch weights.
  - Its single "available ceiling" probe (predictability_r2, full-batch Adam
    on ~1000 sequences) reported R²=-0.04 — underfit at that scale, not a
    real ceiling. This version reports TWO probes and warns if they disagree:
      probe_matched: C9's exact protocol (24 train seqs, 300 full-batch
        steps) — a like-for-like number against the C9 headline.
      probe_full: minibatched Adam over ALL train sequences for
        `c11.probe_epochs` epochs (src.offline_gru.probe_r2_minibatch) — the
        full-batch probe converges at this scale only with minibatching.

Also reports the teacher-forced inter-layer consistency error e1 for both
models and for the true (settled) codes themselves — the only honest
cross-layer signal per the C12 findings.

    python experiments/c11_offline_gru.py --seeds 0 1 2 --device mps
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    parse_args,
    save_temporal_checkpoint,
    setup,
)
from src.metrics import stack_mean_std
from src.offline_gru import (
    cache_eval_codes,
    encode_eval_codes,
    gru_r2_per_layer,
    probe_r2_minibatch,
    teacher_forced_e1,
    train_offline_gru_full,
    true_codes_e1,
)
from src.predictability import predictability_r2
from src.temporal import CoupledTopDownRNN
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def _get_codes(split_name, seqs, deconvs, r_init, cfg, root, seed, use_cache):
    """C11/C12/C13 must share identical codes: go through cache_eval_codes
    for real runs; --smoke never reads/writes the cache."""
    if use_cache:
        return cache_eval_codes(split_name, seqs, deconvs, r_init, cfg, root, seed)
    return encode_eval_codes(seqs, deconvs, r_init, cfg)


def _build_coupled(cfg, r_init, deconvs, device):
    t = cfg["temporal"]
    return CoupledTopDownRNN(
        r_init,
        deconvs,
        delta_scale=t.get("delta_scale", 1.0),
        delta_bounded=t.get("delta_bounded", True),
    ).to(device)


def _dig(d, path):
    cur = d
    for k in path:
        cur = cur[k]
    return cur


def _agg(per_seed, path):
    return mean_std([_dig(ps, path) for ps in per_seed])


def _run_seed(seed, cfg, root, device, c11, c9, use_cache, run_dir, epochs, lr, probe_epochs):
    seed_everything(seed)
    cfg["seed"] = seed
    train, val, test, info = load_data(cfg, root, seed)
    print(f"seed={seed}  {info['n_train']} train / {info['n_test']} test  hash={info['split_hash']}")
    deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

    n_train_seq = min(len(train), c11.get("n_train_sequences", cfg["data"]["n_train"]))
    n_test_seq = min(len(test), c11.get("n_test_sequences", cfg["data"]["n_test"]))
    train_seqs = [s.to(device) for s in train[:n_train_seq]]
    test_seqs = [s.to(device) for s in test[:n_test_seq]]
    print(f"[seed {seed}] encoding eval-protocol codes  n_train={len(train_seqs)}  n_test={len(test_seqs)}")
    train_codes = _get_codes("train", train_seqs, deconvs, r_init, cfg, root, seed, use_cache)
    test_codes = _get_codes("test", test_seqs, deconvs, r_init, cfg, root, seed, use_cache)

    # --- two ceiling probes ---
    probe_model = "conv" if "conv" in c9.get("models", ["conv"]) else c9.get("models", ["linear"])[0]
    n_matched = min(len(train_codes), c9.get("n_train_sequences", 24))
    n_matched_test = min(len(test_codes), cfg["eval"].get("n_pair_sequences", 12))
    probe_matched = predictability_r2(
        train_codes[:n_matched],
        test_codes[:n_matched_test],
        device,
        context=c9.get("context", 2),
        steps=c9.get("steps", 300),
        model=probe_model,
        seed=seed,
    )
    probe_full = probe_r2_minibatch(
        train_codes,
        test_codes,
        device,
        context=c9.get("context", 2),
        epochs=probe_epochs,
        model=probe_model,
        seed=seed,
    )
    diff = abs(probe_matched["r2_vs_copy_last"] - probe_full["best_test_r2"])
    print(
        f"[seed {seed}] probe matched({probe_model}, n_train={n_matched})="
        f"{probe_matched['r2_vs_copy_last']:.4f}  full(best over {probe_epochs} epochs)="
        f"{probe_full['best_test_r2']:.4f}  |diff|={diff:.4f}"
    )
    if diff > 0.1:
        print(
            f"WARNING: seed={seed}: probe_matched ({probe_matched['r2_vs_copy_last']:.3f}) and "
            f"probe_full ({probe_full['best_test_r2']:.3f}) disagree by {diff:.3f} (>0.1) — "
            "the two ceiling estimates are not consistent; do not quote either alone."
        )

    # --- true-code inter-layer error: no model, just the settled codes ---
    true_e1 = true_codes_e1(test_codes, deconvs)

    # --- independent model: full epochs, no early stop, best-epoch weights ---
    indep = build_temporal(cfg, r_init, device)
    indep_res = train_offline_gru_full(indep, train_codes, test_codes, epochs=epochs, lr=lr, device=device, log=print)
    indep.load_state_dict(indep_res["best_state"])
    indep_e1 = teacher_forced_e1(indep, test_codes, deconvs)

    # --- coupled model: same protocol ---
    coupled = _build_coupled(cfg, r_init, deconvs, device)
    coupled_res = train_offline_gru_full(
        coupled, train_codes, test_codes, epochs=epochs, lr=lr, device=device, log=print
    )
    coupled.load_state_dict(coupled_res["best_state"])
    coupled_e1 = teacher_forced_e1(coupled, test_codes, deconvs)
    assert coupled_e1 < 1e-6, (
        f"seed={seed}: coupled model's teacher-forced e1={coupled_e1:.3e} is not ~0 — top-down "
        "generation is broken (r_pred[0] must equal f_clamp(deconv1(r_pred[1])) exactly)."
    )
    coupled_layers = gru_r2_per_layer(coupled, test_codes)["per_layer"]
    layer0_r2 = next(p["r2_vs_copy_last"] for p in coupled_layers if p["layer"] == 0)
    layer1_r2 = next(p["r2_vs_copy_last"] for p in coupled_layers if p["layer"] == 1)

    print(
        f"[seed {seed}] indep best_r2={indep_res['best_r2']:.4f} (epoch {indep_res['best_epoch']})  "
        f"coupled best_r2={coupled_res['best_r2']:.4f} (epoch {coupled_res['best_epoch']})  "
        f"coupled layer0_r2={layer0_r2:.4f} layer1_r2={layer1_r2:.4f}  "
        f"TF e1 indep={indep_e1:.4f} coupled={coupled_e1:.2e} true={true_e1:.4f}"
    )

    checkpoints = {}
    if run_dir is not None:
        indep_ckpt = run_dir / f"temporal_seed{seed}.pt"
        coupled_ckpt = run_dir / f"temporal_coupled_seed{seed}.pt"
        save_temporal_checkpoint(indep_ckpt, indep, cfg, seed=seed)
        save_temporal_checkpoint(coupled_ckpt, coupled, cfg, seed=seed)
        checkpoints = {"independent": str(indep_ckpt), "coupled": str(coupled_ckpt)}

    return {
        "seed": seed,
        "split_hash": info["split_hash"],
        "n_train_sequences": len(train_seqs),
        "n_test_sequences": len(test_seqs),
        "probe_model": probe_model,
        "probe_matched": probe_matched,
        "probe_full": probe_full,
        "probe_disagreement": diff,
        "true_e1": true_e1,
        "independent": {
            "history": indep_res["history"],
            "best_r2": indep_res["best_r2"],
            "best_epoch": indep_res["best_epoch"],
            "final_r2": indep_res["final_r2"],
            "e1": indep_e1,
        },
        "coupled": {
            "history": coupled_res["history"],
            "best_r2": coupled_res["best_r2"],
            "best_epoch": coupled_res["best_epoch"],
            "final_r2": coupled_res["final_r2"],
            "e1": coupled_e1,
            "layer0_r2": layer0_r2,
            "layer1_r2": layer1_r2,
        },
        "checkpoints": checkpoints,
    }


def main():
    args = parse_args("C11 offline ConvGRU: independent vs coupled top-down, on eval-protocol codes")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c11_offline_gru", arm=arm_name(cfg))

    c11 = cfg.get("c11", {})
    c9 = cfg.get("c9", {})
    epochs = args.epochs if args.epochs is not None else c11.get("epochs", 30)
    lr = c11.get("lr", 1e-3)
    probe_epochs = c11.get("probe_epochs", 20)
    use_cache = not args.smoke

    per_seed = []
    for seed in seeds:
        rec = _run_seed(seed, cfg, root, device, c11, c9, use_cache, run_dir, epochs, lr, probe_epochs)
        per_seed.append(rec)
        if not args.smoke:
            art_indep = Path(root) / "artifacts" / f"offline_gru_seed{seed}.pt"
            art_coupled = Path(root) / "artifacts" / f"offline_gru_coupled_seed{seed}.pt"
            art_indep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec["checkpoints"]["independent"], art_indep)
            shutil.copy2(rec["checkpoints"]["coupled"], art_coupled)
            print(f"[seed {seed}] saved {art_indep}  {art_coupled}")
        else:
            print(f"[seed {seed}] smoke: skipped artifacts/ copy")

    # --- aggregate across seeds ---
    indep_best_mu, indep_best_sd = _agg(per_seed, ["independent", "best_r2"])
    indep_epoch_mu, indep_epoch_sd = _agg(per_seed, ["independent", "best_epoch"])
    indep_final_mu, indep_final_sd = _agg(per_seed, ["independent", "final_r2"])
    indep_e1_mu, indep_e1_sd = _agg(per_seed, ["independent", "e1"])

    coup_best_mu, coup_best_sd = _agg(per_seed, ["coupled", "best_r2"])
    coup_epoch_mu, coup_epoch_sd = _agg(per_seed, ["coupled", "best_epoch"])
    coup_final_mu, coup_final_sd = _agg(per_seed, ["coupled", "final_r2"])
    coup_e1_mu, coup_e1_sd = _agg(per_seed, ["coupled", "e1"])
    coup_l0_mu, coup_l0_sd = _agg(per_seed, ["coupled", "layer0_r2"])
    coup_l1_mu, coup_l1_sd = _agg(per_seed, ["coupled", "layer1_r2"])

    probe_matched_mu, probe_matched_sd = _agg(per_seed, ["probe_matched", "r2_vs_copy_last"])
    probe_full_mu, probe_full_sd = _agg(per_seed, ["probe_full", "best_test_r2"])
    true_e1_mu, true_e1_sd = _agg(per_seed, ["true_e1"])

    indep_curve_mu, indep_curve_sd = stack_mean_std(
        [[h["test_r2"] for h in ps["independent"]["history"]] for ps in per_seed]
    )
    coup_curve_mu, coup_curve_sd = stack_mean_std(
        [[h["test_r2"] for h in ps["coupled"]["history"]] for ps in per_seed]
    )

    agg_diff = abs(probe_matched_mu - probe_full_mu)
    if agg_diff > 0.1:
        print(
            f"WARNING: mean probe_matched={probe_matched_mu:.3f} and mean probe_full="
            f"{probe_full_mu:.3f} disagree by {agg_diff:.3f} (>0.1) across seeds."
        )

    summary = (
        f"C11 | GRU R² best={indep_best_mu:.3f}±{indep_best_sd:.3f} (epoch {indep_epoch_mu:.1f}) "
        f"coupled={coup_best_mu:.3f}±{coup_best_sd:.3f} (epoch {coup_epoch_mu:.1f}) | "
        f"probe matched={probe_matched_mu:.3f} (C9 ref) full={probe_full_mu:.3f} | "
        f"TF e1 indep={indep_e1_mu:.4f} coupled={coup_e1_mu:.2e} true={true_e1_mu:.4f} | "
        f"seeds={len(seeds)}"
    )

    metrics = {
        "claim": "C11",
        "seeds": per_seed,
        "epochs": epochs,
        "probe_epochs": probe_epochs,
        "lr": lr,
        "probe_model": per_seed[0]["probe_model"] if per_seed else None,
        "independent": {
            "history_mean": indep_curve_mu,
            "history_std": indep_curve_sd,
            "best_r2_mean": indep_best_mu,
            "best_r2_std": indep_best_sd,
            "best_epoch_mean": indep_epoch_mu,
            "best_epoch_std": indep_epoch_sd,
            "final_r2_mean": indep_final_mu,
            "final_r2_std": indep_final_sd,
            "e1_mean": indep_e1_mu,
            "e1_std": indep_e1_sd,
        },
        "coupled": {
            "history_mean": coup_curve_mu,
            "history_std": coup_curve_sd,
            "best_r2_mean": coup_best_mu,
            "best_r2_std": coup_best_sd,
            "best_epoch_mean": coup_epoch_mu,
            "best_epoch_std": coup_epoch_sd,
            "final_r2_mean": coup_final_mu,
            "final_r2_std": coup_final_sd,
            "e1_mean": coup_e1_mu,
            "e1_std": coup_e1_sd,
            "layer0_r2_mean": coup_l0_mu,
            "layer0_r2_std": coup_l0_sd,
            "layer1_r2_mean": coup_l1_mu,
            "layer1_r2_std": coup_l1_sd,
        },
        "probe_matched_r2_mean": probe_matched_mu,
        "probe_matched_r2_std": probe_matched_sd,
        "probe_full_best_r2_mean": probe_full_mu,
        "probe_full_best_r2_std": probe_full_sd,
        "true_e1_mean": true_e1_mu,
        "true_e1_std": true_e1_sd,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
