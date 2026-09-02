#!/usr/bin/env python3
"""C7 — slowness regularization cannot fix smoothness in-place (λ_slow × unroll_k).

`train_loop` (src/temporal.py) adds `lambda_slow * mean((r_curr - r_prev1)**2)`
to the weight loss. With the *detached* settle (`settle_with_temporal_prior`),
`r_curr` is a detached leaf by the time that term is built, so it has ZERO
gradient into the dictionary or the GRU — any "plateau" that measurement
finds is guaranteed by construction, not evidence about slowness itself.

This sweep crosses λ_slow with `unroll_k`: for unroll_k=0 the settle is the
old detached one (no gradient path, by construction); for unroll_k>0 the last
`unroll_k` settle iterations are NOT detached
(`settle_with_temporal_prior_unrolled`), so the slowness term (and the
temporal-prior pull) actually reach the dictionary / GRU. Comparing the two
answers whether giving slowness a real gradient path changes anything.

For every (λ_slow, unroll_k) cell we train with the dictionary UNFROZEN (as
before), then measure post-training smoothness two ways:
  - no_pull:   settle_grounded, warm-start, no slowness term at all — isolates
               whether the DICTIONARY itself changed shape.
  - with_pull: settle_with_temporal_prior, warm-start, r_pred=r_prev1 and the
               cell's λ_slow — this is what the actual train/eval pipeline
               sees (settle pulled toward the previous frame's code).
We also record the relative Frobenius drift of each deconv weight vs the
pretrained checkpoint (dictionary changed at all?), the teacher-forced
val_mse / copy_last_mse at the end of training (did slowness hurt fitting?),
and whether `slowness_has_dictionary_grad` finds a nonzero gradient path for
that unroll_k (documents the mechanism directly, not just its downstream
effect).
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    arm_name,
    build_temporal,
    ensure_dictionary,
    load_data,
    parse_args,
    setup,
    train_temporal_pc,
)
from src.inference import (
    collect_unrelated_frames,
    diagnose_latent_smoothness,
    settle_grounded,
    settle_with_temporal_prior,
)
from src.metrics import pair_stats
from src.spatial_pc import unfreeze_dictionary
from src.temporal import slowness_has_dictionary_grad
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def lambdas(cfg):
    return cfg.get("c7", {}).get("lambda_slow", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])


def unroll_ks(cfg):
    return cfg.get("c7", {}).get("unroll_k", [0, 5])


def diagnose_pull_smoothness(
    seq,
    unrelated_frames,
    r_init,
    deconvs,
    alpha,
    lr_r,
    sigma_2,
    num_epochs_inner,
    num_layers,
    lambda_slow,
    temporal_prior_weight,
    max_unrelated_pairs=40,
    init_noise=0.01,
    use_prior=True,
    log=print,
):
    """Same cons/unrelated cosine measurement as diagnose_latent_smoothness,
    but the trajectory is settled with settle_with_temporal_prior (r_pred set
    to r_prev1, this cell's λ_slow) instead of plain settle_grounded — this is
    what the training/eval pipeline actually sees at inference time, vs
    `no_pull` which isolates whether the dictionary itself changed.
    """
    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]
    rs = []
    r_prev = None
    for t in range(T):
        I = seq[t]
        if r_prev is None:
            r_t, _ = settle_grounded(
                I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
                init_noise=init_noise, use_prior=use_prior,
            )
        else:
            r0 = [ri.detach().clone() for ri in r_prev]
            r_t, _, _ = settle_with_temporal_prior(
                I, r0, r_prev, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
                r_prev1=r_prev, lambda_slow=lambda_slow, use_prior=use_prior,
                temporal_prior_weight=temporal_prior_weight,
            )
            r_t = [ri.detach().clone() for ri in r_t]
        rs.append(r_t)
        r_prev = r_t

    cons_cos, cons_rel = [], []
    for t in range(1, T):
        st = pair_stats(rs[t], rs[t - 1])
        cons_cos.append(st["cos"])
        cons_rel.append(st["rel"])

    u_rs = [
        settle_grounded(
            I, r_init, deconvs, alpha, lr_r, sigma_2, num_epochs_inner, num_layers,
            init_noise=init_noise, use_prior=use_prior,
        )[0]
        for I in unrelated_frames
    ]
    n_u = len(u_rs)
    pairs = [(i, j) for i in range(n_u) for j in range(i + 1, n_u)]
    if len(pairs) > max_unrelated_pairs:
        idx = torch.randperm(len(pairs))[:max_unrelated_pairs].tolist()
        pairs = [pairs[k] for k in idx]
    un_cos, un_rel = [], []
    for i, j in pairs:
        st = pair_stats(u_rs[i], u_rs[j])
        un_cos.append(st["cos"])
        un_rel.append(st["rel"])

    m_cc = float(np.mean(cons_cos)) if cons_cos else float("nan")
    m_cr = float(np.mean(cons_rel)) if cons_rel else float("nan")
    m_uc = float(np.mean(un_cos)) if un_cos else float("nan")
    m_ur = float(np.mean(un_rel)) if un_rel else float("nan")
    frac_rel = m_cr / (m_ur + 1e-8)
    log(f"  [with_pull λ={lambda_slow}] cons cos={m_cc:.4f} frac_rel={frac_rel:.4f}")
    return {"cons_cos": m_cc, "cons_rel": m_cr, "frac_rel": frac_rel, "un_cos": m_uc, "un_rel": m_ur}


def dictionary_drift(deconvs, ref_weights):
    drifts = []
    for d, ref in zip(deconvs, ref_weights):
        num = torch.norm(d.weight.detach() - ref).item()
        den = torch.norm(ref).item() + 1e-8
        drifts.append(num / den)
    return float(np.mean(drifts)) if drifts else float("nan")


def main():
    args = parse_args("C7 slowness sweep (λ_slow × unroll_k)")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c7_slowness_sweep", arm=arm_name(cfg))
    inf = cfg["inference"]
    sweep = []
    cfg = copy.deepcopy(cfg)
    cfg["spatial"]["freeze"] = False

    for unroll_k in unroll_ks(cfg):
        for lam in lambdas(cfg):
            cons_no_pull, frac_no_pull, cons_pull, frac_pull = [], [], [], []
            drift_s, val_mse_s, copy_last_s, grad_path_s = [], [], [], []
            for seed in seeds:
                seed_everything(seed)
                cell_cfg = copy.deepcopy(cfg)
                cell_cfg["temporal"]["lambda_slow"] = lam
                cell_cfg["temporal"]["slow_unroll_k"] = unroll_k
                cell_cfg["seed"] = seed
                cell_cfg["spatial"]["freeze"] = False
                train, val, test, info = load_data(cell_cfg, root, seed)
                print(f"unroll_k={unroll_k} lambda_slow={lam} seed={seed}  {info['n_train']} train")
                deconvs, r_init, _ = ensure_dictionary(cell_cfg, train, device, root)
                unfreeze_dictionary(deconvs)
                ref_weights = [d.weight.detach().clone() for d in deconvs]
                model = build_temporal(cell_cfg, r_init, device)

                grad_path_s.append(
                    slowness_has_dictionary_grad(
                        train[0][0].to(device),
                        r_init,
                        deconvs,
                        model,
                        inf["alpha"],
                        inf["lr_r"],
                        inf["sigma_2"],
                        inf["num_epochs_inner"],
                        cfg["spatial"]["num_layers"],
                        lambda_slow=max(lam, 1.0),
                        temporal_prior_weight=inf.get("temporal_prior_weight", 0.01),
                        slow_unroll_k=unroll_k,
                    )
                )

                model, history, _ = train_temporal_pc(
                    train, r_init, deconvs, model, cell_cfg, device, val_seq=val[0], log=print
                )
                if history:
                    val_mse_s.append(history[-1].get("val_mse", float("nan")))
                    copy_last_s.append(history[-1].get("copy_last_mse", float("nan")))
                drift_s.append(dictionary_drift(deconvs, ref_weights))

                unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
                n_seq = cfg["eval"]["n_pair_sequences"]
                seq_cos_np, seq_frac_np, seq_cos_wp, seq_frac_wp = [], [], [], []
                for seq in test[:n_seq]:
                    sm = diagnose_latent_smoothness(
                        seq.to(device), unrelated, r_init, deconvs,
                        alpha=inf["alpha"], lr_r=inf["lr_r"], sigma_2=inf["sigma_2"],
                        num_epochs_inner=inf["num_epochs_inner"], num_layers=cfg["spatial"]["num_layers"],
                        max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"], warm_start=True,
                        label=f"no_pull λ={lam} k={unroll_k}", init_noise=inf.get("init_noise", 0.01),
                        use_prior=inf.get("use_prior", True), log=print,
                    )
                    seq_cos_np.append(sm["cons_cos"])
                    seq_frac_np.append(sm["frac_rel"])

                    wp = diagnose_pull_smoothness(
                        seq.to(device), unrelated, r_init, deconvs,
                        inf["alpha"], inf["lr_r"], inf["sigma_2"], inf["num_epochs_inner"],
                        cfg["spatial"]["num_layers"], lam, inf.get("temporal_prior_weight", 0.01),
                        max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
                        init_noise=inf.get("init_noise", 0.01), use_prior=inf.get("use_prior", True), log=print,
                    )
                    seq_cos_wp.append(wp["cons_cos"])
                    seq_frac_wp.append(wp["frac_rel"])

                cons_no_pull.append(sum(seq_cos_np) / max(len(seq_cos_np), 1))
                frac_no_pull.append(sum(seq_frac_np) / max(len(seq_frac_np), 1))
                cons_pull.append(sum(seq_cos_wp) / max(len(seq_cos_wp), 1))
                frac_pull.append(sum(seq_frac_wp) / max(len(seq_frac_wp), 1))

            mu_np, sd_np = mean_std(cons_no_pull)
            fmu_np, fsd_np = mean_std(frac_no_pull)
            mu_wp, sd_wp = mean_std(cons_pull)
            fmu_wp, fsd_wp = mean_std(frac_pull)
            drift_mu, _ = mean_std(drift_s)
            val_mu, val_sd = mean_std(val_mse_s)
            cl_mu, cl_sd = mean_std(copy_last_s)
            grad_path = bool(any(grad_path_s))
            sweep.append({
                "lambda_slow": lam,
                "unroll_k": unroll_k,
                "cons_cos_no_pull": mu_np,
                "cons_cos_no_pull_std": sd_np,
                "frac_rel_no_pull": fmu_np,
                "frac_rel_no_pull_std": fsd_np,
                "cons_cos_with_pull": mu_wp,
                "cons_cos_with_pull_std": sd_wp,
                "frac_rel_with_pull": fmu_wp,
                "frac_rel_with_pull_std": fsd_wp,
                "dict_drift": drift_mu,
                "val_mse": val_mu,
                "val_mse_std": val_sd,
                "copy_last_mse": cl_mu,
                "copy_last_mse_std": cl_sd,
                "grad_path": grad_path,
                "per_seed": {
                    "seeds": list(seeds),
                    "cons_cos_no_pull": cons_no_pull,
                    "frac_rel_no_pull": frac_no_pull,
                    "cons_cos_with_pull": cons_pull,
                    "frac_rel_with_pull": frac_pull,
                    "dict_drift": drift_s,
                    "val_mse": val_mse_s,
                    "copy_last_mse": copy_last_s,
                    "grad_path": grad_path_s,
                },
                # kept for backward-compatible readers expecting the pre-rework keys
                "cons_cos_mean": mu_np,
                "cons_cos_std": sd_np,
                "frac_rel_mean": fmu_np,
                "frac_rel_std": fsd_np,
            })
            print(
                f"C7 | k={unroll_k} λ_slow={lam}  no_pull cos={mu_np:.4f}±{sd_np:.4f}  "
                f"with_pull cos={mu_wp:.4f}±{sd_wp:.4f}  drift={drift_mu:.4f}  grad_path={grad_path}"
            )

    def _span(unroll_k, key):
        vals = [s[key] for s in sweep if s["unroll_k"] == unroll_k]
        return (max(vals) - min(vals)) if vals else float("nan")

    ks = sorted(set(s["unroll_k"] for s in sweep))
    k0 = ks[0] if ks else 0
    k_hi = ks[-1] if len(ks) > 1 else k0
    plateau_span_k0 = _span(k0, "cons_cos_no_pull")
    plateau_span_k5 = _span(k_hi, "cons_cos_no_pull")

    def _range_str(unroll_k):
        vals = sorted(
            [(s["lambda_slow"], s["cons_cos_no_pull"]) for s in sweep if s["unroll_k"] == unroll_k],
            key=lambda t: t[0],
        )
        if not vals:
            return "n/a"
        return f"{vals[0][1]:.3f}..{vals[-1][1]:.3f}"

    def _drift_str(unroll_k):
        vals = [s["dict_drift"] for s in sweep if s["unroll_k"] == unroll_k]
        return f"{np.mean(vals):.4f}" if vals else "n/a"

    summary = (
        f"C7 | no gradient path (k={k0}): cons cos λ=0..{lambdas(cfg)[-1]} → {_range_str(k0)} "
        f"(span {plateau_span_k0:.3f}) | with path (k={k_hi}): → {_range_str(k_hi)} "
        f"(span {plateau_span_k5:.3f}) | dict drift k0={_drift_str(k0)} k{k_hi}={_drift_str(k_hi)}"
    )
    metrics = {
        "claim": "C7",
        "sweep": sweep,
        "plateau_span_k0": plateau_span_k0,
        "plateau_span_k5": plateau_span_k5,
        "note": (
            "unroll_k=0 uses settle_with_temporal_prior, which detaches r_curr every "
            "step, so lambda_slow in the weight loss has zero gradient into the "
            "dictionary or GRU (verified directly by grad_path=False via "
            "slowness_has_dictionary_grad). unroll_k>0 uses "
            "settle_with_temporal_prior_unrolled, which leaves the last unroll_k "
            "steps un-detached, giving that term a real gradient path "
            "(grad_path=True). no_pull isolates whether the dictionary itself "
            "changed (settle_grounded, no slowness term); with_pull is what the "
            "actual train/eval pipeline sees (settle_with_temporal_prior, "
            "r_pred=r_prev1, this cell's lambda_slow)."
        ),
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
