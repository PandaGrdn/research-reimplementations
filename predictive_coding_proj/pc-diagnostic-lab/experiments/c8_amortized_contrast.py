#!/usr/bin/env python3
"""C8 — amortized encoder removes the noise floor; smoothness and rollout improve."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.amortized import (
    amortized_same_frame_cos,
    amortized_short_rollout,
    amortized_smoothness,
    build_ae,
    train_ae,
    train_amortized_temporal,
)
from src.experiment import (
    build_temporal,
    ensure_dictionary,
    eval_long_rollouts,
    load_data,
    parse_args,
    setup,
    train_temporal_pc,
)
from src.inference import collect_eval_frames, collect_unrelated_frames, diagnose_latent_smoothness
from src.rollout import validate_hierarchical
from src.temporal import TemporalConvRNN
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def iterative_arm(cfg, train, val, test, device, root, seed):
    inf = cfg["inference"]
    deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)
    unrelated = [fr.to(device) for fr in collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])]
    n_seq = cfg["eval"]["n_pair_sequences"]
    sm = diagnose_latent_smoothness(
        test[0].to(device),
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
        label="iterative",
        init_noise=inf.get("init_noise", 0.01),
        use_prior=inf.get("use_prior", True),
        log=print,
    )
    extra_cons, extra_un = list(sm["cons_cos_list"]), list(sm["un_cos_list"])
    extra_rel, extra_un_rel = [sm["cons_rel"]], [sm["un_rel"]]
    for seq in test[1:n_seq]:
        more = diagnose_latent_smoothness(
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
            label="iterative",
            init_noise=inf.get("init_noise", 0.01),
            use_prior=inf.get("use_prior", True),
            log=lambda *_: None,
        )
        extra_cons.extend(more["cons_cos_list"])
        extra_un.extend(more["un_cos_list"])
        extra_rel.append(more["cons_rel"])
        extra_un_rel.append(more["un_rel"])
    sm["cons_cos_list"] = extra_cons
    sm["un_cos_list"] = extra_un
    sm["n_cons"] = len(extra_cons)
    sm["cons_cos"] = sum(extra_cons) / max(len(extra_cons), 1)
    sm["un_cos"] = sum(extra_un) / max(len(extra_un), 1)
    mean_cons_rel = sum(extra_rel) / max(len(extra_rel), 1)
    mean_un_rel = sum(extra_un_rel) / max(len(extra_un_rel), 1)
    sm["frac_rel"] = mean_cons_rel / (mean_un_rel + 1e-8)

    model = build_temporal(cfg, r_init, device)
    model, _, last = train_temporal_pc(
        train, r_init, deconvs, model, cfg, device, val_seq=val[0], log=print
    )
    short = validate_hierarchical(
        test[0].to(device),
        r_init,
        inf["num_epochs_inner"],
        cfg["spatial"]["num_layers"],
        inf["sigma_2"],
        inf["alpha"],
        inf["lr_r"],
        deconvs,
        model,
        use_prior=inf.get("use_prior", True),
        log=print,
    )
    headline_sp = cfg["eval"].get("headline_split", 10)
    long = eval_long_rollouts(
        test, r_init, deconvs, model, cfg, device, split_points=[headline_sp], log=print
    )
    return {
        "smoothness": {
            "cons_cos": sm["cons_cos"],
            "un_cos": sm["un_cos"],
            "frac_rel": sm["frac_rel"],
            "cons_cos_list": extra_cons,
            "un_cos_list": extra_un,
            "n_cons": len(extra_cons),
        },
        "same_frame_cos": None,
        "short_mse": short["mse"],
        "long_mse": long[str(headline_sp)]["long_mse_mean"],
    }


def amortized_arm(cfg, train, val, test, device, seed):
    ae = build_ae(cfg).to(device)
    train_ae(ae, train, cfg, device, log=print)
    frames = [fr.to(device) for fr in collect_eval_frames(test, cfg["eval"]["n_determinism_frames"])]
    same_cos, _ = amortized_same_frame_cos(ae, frames, device)
    unrelated = collect_unrelated_frames(test, cfg["eval"]["n_unrelated_frames"])
    n_seq = cfg["eval"]["n_pair_sequences"]
    sm = amortized_smoothness(
        ae,
        [s.to(device) for s in test[:n_seq]],
        unrelated,
        device,
        max_unrelated_pairs=cfg["eval"]["max_unrelated_pairs"],
    )
    r_probe = ae.encode(test[0][:1].to(device))
    temporal_nn = TemporalConvRNN(
        [r_probe],
        delta_scale=cfg["temporal"].get("delta_scale", 1.0),
        delta_bounded=cfg["temporal"].get("delta_bounded", True),
    ).to(device)
    train_amortized_temporal(ae, temporal_nn, train, cfg, device, log=print)
    short = amortized_short_rollout(ae, temporal_nn, test[0], device, context=2)
    return {
        "smoothness": sm,
        "same_frame_cos": same_cos,
        "short_mse": short["mse"],
        "mse_per_frame": short["mse_per_frame"],
    }


def main():
    args = parse_args("C8 amortized contrast")
    root, cfg, device, seeds = setup(args)
    run_dir = new_run_dir(root, "c8_amortized_contrast")
    it_cos, am_cos, it_mse, am_mse, same = [], [], [], [], []
    pooled = {"iterative": {"cons": [], "un": []}, "amortized": {"cons": [], "un": []}}
    per_seed = []

    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_train']} train  hash={info['split_hash']}")
        it = iterative_arm(cfg, train, val, test, device, root, seed)
        am = amortized_arm(cfg, train, val, test, device, seed)
        it_cos.append(it["smoothness"]["cons_cos"])
        am_cos.append(am["smoothness"]["cons_cos"])
        it_mse.append(it["short_mse"])
        am_mse.append(am["short_mse"])
        same.append(am["same_frame_cos"])
        pooled["iterative"]["cons"].extend(it["smoothness"]["cons_cos_list"])
        pooled["iterative"]["un"].extend(it["smoothness"]["un_cos_list"])
        pooled["amortized"]["cons"].extend(am["smoothness"]["cons_cos_list"])
        pooled["amortized"]["un"].extend(am["smoothness"]["un_cos_list"])
        per_seed.append({"seed": seed, "iterative": it, "amortized": am, "split_hash": info["split_hash"]})

    summary = (
        f"C8 | same-frame cos {mean_std(same)[0]:.3f}  "
        f"consec cos iter={mean_std(it_cos)[0]:.3f} vs amort={mean_std(am_cos)[0]:.3f}  "
        f"short MSE iter={mean_std(it_mse)[0]:.4f} vs amort={mean_std(am_mse)[0]:.4f}"
    )
    metrics = {
        "claim": "C8",
        "seeds": per_seed,
        "iterative": {
            "cons_cos_mean": mean_std(it_cos)[0],
            "cons_cos_std": mean_std(it_cos)[1],
            "short_mse_mean": mean_std(it_mse)[0],
            "short_mse_std": mean_std(it_mse)[1],
            "cons_cos_list": pooled["iterative"]["cons"],
            "un_cos_list": pooled["iterative"]["un"],
        },
        "amortized": {
            "same_frame_cos_mean": mean_std(same)[0],
            "same_frame_cos_std": mean_std(same)[1],
            "cons_cos_mean": mean_std(am_cos)[0],
            "cons_cos_std": mean_std(am_cos)[1],
            "short_mse_mean": mean_std(am_mse)[0],
            "short_mse_std": mean_std(am_mse)[1],
            "cons_cos_list": pooled["amortized"]["cons"],
            "un_cos_list": pooled["amortized"]["un"],
        },
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
