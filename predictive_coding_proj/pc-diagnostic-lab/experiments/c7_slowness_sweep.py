#!/usr/bin/env python3
"""C7 — slowness regularization cannot fix smoothness in-place (λ_slow sweep)."""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import (
    build_temporal,
    ensure_dictionary,
    load_data,
    overlay_temporal,
    parse_args,
    setup,
    train_temporal_pc,
)
from src.inference import collect_unrelated_frames, diagnose_latent_smoothness
from src.spatial_pc import unfreeze_dictionary
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def lambdas(cfg):
    return cfg.get("c7", {}).get("lambda_slow", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0])


def main():
    args = parse_args("C7 slowness sweep")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c7_slowness_sweep")
    inf = cfg["inference"]
    sweep = []
    cfg = copy.deepcopy(cfg)
    cfg["spatial"]["freeze"] = False

    for lam in lambdas(cfg):
        cons_s, frac_s = [], []
        for seed in seeds:
            seed_everything(seed)
            cell_cfg = overlay_temporal(cfg, lambda_slow=lam)
            cell_cfg["seed"] = seed
            cell_cfg["spatial"]["freeze"] = False
            train, val, test, info = load_data(cell_cfg, root, seed)
            print(f"lambda_slow={lam} seed={seed}  {info['n_train']} train")
            deconvs, r_init, _ = ensure_dictionary(cell_cfg, train, device, root)
            unfreeze_dictionary(deconvs)
            model = build_temporal(cell_cfg, r_init, device)
            model, _, _ = train_temporal_pc(
                train, r_init, deconvs, model, cell_cfg, device, val_seq=val[0], log=print
            )
            unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
            n_seq = cfg["eval"]["n_pair_sequences"]
            seq_cos, seq_frac = [], []
            for seq in test[:n_seq]:
                sm = diagnose_latent_smoothness(
                    seq.to(device),
                    unrelated,
                    r_init,
                    deconvs,
                    alpha=inf["alpha"],
                    lr_r=inf["lr_r"],
                    sigma_2=inf["sigma_2"],
                    num_epochs_inner=inf["num_epochs_inner"],
                    num_layers=cfg["spatial"]["num_layers"],
                    max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
                    warm_start=True,
                    label=f"λ_slow={lam}",
                    init_noise=inf.get("init_noise", 0.01),
                    use_prior=inf.get("use_prior", True),
                    log=print,
                )
                seq_cos.append(sm["cons_cos"])
                seq_frac.append(sm["frac_rel"])
            cons_s.append(sum(seq_cos) / max(len(seq_cos), 1))
            frac_s.append(sum(seq_frac) / max(len(seq_frac), 1))
        mu, sd = mean_std(cons_s)
        fmu, fsd = mean_std(frac_s)
        sweep.append({
            "lambda_slow": lam,
            "cons_cos_mean": mu,
            "cons_cos_std": sd,
            "frac_rel_mean": fmu,
            "frac_rel_std": fsd,
            "per_seed_cons_cos": cons_s,
        })
        print(f"C7 | λ_slow={lam}  consec cos={mu:.4f} ± {sd:.4f}")

    plateau = max(s["cons_cos_mean"] for s in sweep) - min(s["cons_cos_mean"] for s in sweep)
    summary = (
        "C7 | consec cos vs λ_slow: "
        + "  ".join(f"{s['lambda_slow']}={s['cons_cos_mean']:.3f}" for s in sweep)
        + f"  plateau_span={plateau:.3f}"
    )
    metrics = {
        "claim": "C7",
        "sweep": sweep,
        "plateau_span": plateau,
        "note": (
            "Slowness on r has no gradient path into the dictionary: r_curr is a detached "
            "leaf after settle, so λ_slow in the weight loss cannot reshape the codebook. "
            "The only effect is a pull during inference, which plateaus."
        ),
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
