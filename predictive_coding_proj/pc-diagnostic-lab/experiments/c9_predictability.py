#!/usr/bin/env python3
"""C9 — held-out predictability (R^2 vs copy-last) and translation consistency.

C5 shows consecutive-frame code cosine/L2 distance is about as bad as unrelated
frames — but that measures geometry, not predictability, and pixel space is
just as "non-smooth" under the same metric while being perfectly predictable
(translation). C9 asks the sharper question directly: is the code a learnable
function of its own past (vs a copy-last floor), and does it co-translate the
way an equivariant dictionary code should.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import arm_name, ensure_dictionary, load_data, parse_args, setup
from src.inference import collect_eval_frames
from src.predictability import (
    encode_sequences,
    make_settle_encoder,
    pixel_codes,
    predictability_r2,
    translation_consistency,
)
from src.utils import finish_run, mean_std, new_run_dir, seed_everything


def _translation_aggregates(trans):
    per_shift = trans["per_shift"]
    exact_cos = [ly["cos"] for s in per_shift for ly in s["layers"] if ly["exact_expected"]]
    aliased_cos = [ly["cos"] for s in per_shift for ly in s["layers"] if not ly["exact_expected"]]
    pixel_cos = [s["pixel_cos"] for s in per_shift]
    return {
        "per_shift": per_shift,
        "exact_shift_cos_mean": float(np.mean(exact_cos)) if exact_cos else None,
        "aliased_shift_cos_mean": float(np.mean(aliased_cos)) if aliased_cos else None,
        "pixel_cos_mean": float(np.mean(pixel_cos)) if pixel_cos else None,
    }


def _aggregate_predictability(per_seed, models):
    fields = ["mse_pred", "mse_copy_last", "r2_vs_copy_last", "mse_mean_target", "r2_vs_mean"]
    out = {}
    for model in models:
        out[model] = {}
        for space in ("code", "pixel"):
            dicts = [ps["predictability"][model][space] for ps in per_seed]
            agg = {}
            for f in fields:
                mu, sd = mean_std([d[f] for d in dicts])
                agg[f"{f}_mean"] = mu
                agg[f"{f}_std"] = sd
            agg["n_train_windows"] = dicts[0]["n_train_windows"]
            agg["n_test_windows"] = dicts[0]["n_test_windows"]
            agg["context"] = dicts[0]["context"]
            agg["model"] = dicts[0]["model"]
            agg["per_layer"] = dicts[0]["per_layer"]
            out[model][space] = agg
    return out


def _aggregate_translation(per_seed):
    shifts_ref = per_seed[0]["translation"]["per_shift"]
    per_shift_agg = []
    for i, s0 in enumerate(shifts_ref):
        pixel_vals = [ps["translation"]["per_shift"][i]["pixel_cos"] for ps in per_seed]
        pc_mu, pc_sd = mean_std(pixel_vals)
        layers_agg = []
        for l, ly0 in enumerate(s0["layers"]):
            cos_vals = [ps["translation"]["per_shift"][i]["layers"][l]["cos"] for ps in per_seed]
            rel_vals = [ps["translation"]["per_shift"][i]["layers"][l]["rel"] for ps in per_seed]
            c_mu, c_sd = mean_std(cos_vals)
            r_mu, r_sd = mean_std(rel_vals)
            layers_agg.append(
                {
                    "layer": l,
                    "cos_mean": c_mu,
                    "cos_std": c_sd,
                    "rel_mean": r_mu,
                    "rel_std": r_sd,
                    "exact_expected": ly0["exact_expected"],
                    "aliased": ly0["aliased"],
                }
            )
        per_shift_agg.append(
            {"dx": s0["dx"], "dy": s0["dy"], "pixel_cos_mean": pc_mu, "pixel_cos_std": pc_sd, "layers": layers_agg}
        )
    exact_cos = [ly["cos_mean"] for s in per_shift_agg for ly in s["layers"] if ly["exact_expected"]]
    aliased_cos = [ly["cos_mean"] for s in per_shift_agg for ly in s["layers"] if not ly["exact_expected"]]
    pixel_cos = [s["pixel_cos_mean"] for s in per_shift_agg]
    return {
        "per_shift": per_shift_agg,
        "exact_shift_cos_mean": float(np.mean(exact_cos)) if exact_cos else None,
        "aliased_shift_cos_mean": float(np.mean(aliased_cos)) if aliased_cos else None,
        "pixel_cos_mean": float(np.mean(pixel_cos)) if pixel_cos else None,
    }


def main():
    args = parse_args("C9 held-out predictability + translation consistency")
    root, cfg, device, seeds = setup(args, n_seeds_key="n_seeds_inference")
    run_dir = new_run_dir(root, "c9_predictability", arm=arm_name(cfg))

    c9cfg = cfg["c9"]
    n_train_seq = c9cfg["n_train_sequences"]
    n_test_seq = cfg["eval"]["n_pair_sequences"]
    steps = c9cfg["steps"]
    context = c9cfg["context"]
    models = c9cfg["models"]
    shifts = [tuple(s) for s in c9cfg["shifts"]]
    n_translation_frames = c9cfg["n_translation_frames"]

    stride = cfg["spatial"]["stride"]
    num_layers = cfg["spatial"]["num_layers"]
    strides = [stride ** (l + 1) for l in range(num_layers)]

    per_seed = []
    for seed in seeds:
        seed_everything(seed)
        cfg["seed"] = seed
        train, val, test, info = load_data(cfg, root, seed)
        print(f"seed={seed}  {info['n_test']} test  hash={info['split_hash']}")
        deconvs, r_init, _ = ensure_dictionary(cfg, train, device, root)

        train_seqs = [s.to(device) for s in train[:n_train_seq]]
        test_seqs = [s.to(device) for s in test[:n_test_seq]]

        warm_encoder = make_settle_encoder(deconvs, r_init, cfg, warm=True)
        train_codes = encode_sequences(train_seqs, warm_encoder)
        test_codes = encode_sequences(test_seqs, warm_encoder)
        train_pixels = pixel_codes(train_seqs)
        test_pixels = pixel_codes(test_seqs)

        predictability = {}
        for model in models:
            code_r2 = predictability_r2(
                train_codes, test_codes, device, context=context, steps=steps, model=model, seed=seed
            )
            pixel_r2 = predictability_r2(
                train_pixels, test_pixels, device, context=context, steps=steps, model=model, seed=seed
            )
            predictability[model] = {"code": code_r2, "pixel": pixel_r2}
            print(
                f"  seed={seed} model={model}  R2 vs copy-last  code={code_r2['r2_vs_copy_last']:.3f}  "
                f"pixel={pixel_r2['r2_vs_copy_last']:.3f}"
            )

        cold_encoder = make_settle_encoder(deconvs, r_init, cfg, warm=False, init_noise=0.0)
        translation_frames = collect_eval_frames(test_seqs, n_translation_frames)
        translation_frames = [fr.unsqueeze(0) if fr.ndim == 3 else fr for fr in translation_frames]
        trans_raw = translation_consistency(translation_frames, cold_encoder, shifts, strides)
        translation = _translation_aggregates(trans_raw)
        print(
            f"  seed={seed}  translation cos exact={translation['exact_shift_cos_mean']}  "
            f"aliased={translation['aliased_shift_cos_mean']}  pixel={translation['pixel_cos_mean']}"
        )

        per_seed.append(
            {
                "seed": seed,
                "predictability": predictability,
                "translation": translation,
                "split_hash": info["split_hash"],
            }
        )

    predictability_agg = _aggregate_predictability(per_seed, models)
    translation_agg = _aggregate_translation(per_seed)

    primary_model = "conv" if "conv" in models else models[-1]
    code_r2 = predictability_agg[primary_model]["code"]["r2_vs_copy_last_mean"]
    pixel_r2 = predictability_agg[primary_model]["pixel"]["r2_vs_copy_last_mean"]
    n_test_windows = predictability_agg[primary_model]["code"]["n_test_windows"]
    summary = (
        f"C9 | R² vs copy-last ({primary_model}): code={code_r2:.3f} pixel={pixel_r2:.3f} | "
        f"translation cos exact-shift={translation_agg['exact_shift_cos_mean']:.3f} "
        f"aliased-shift={translation_agg['aliased_shift_cos_mean']:.3f} "
        f"(pixel {translation_agg['pixel_cos_mean']:.3f}) | n_test_windows={n_test_windows}"
    )

    metrics = {
        "claim": "C9",
        "seeds": per_seed,
        "predictability": predictability_agg,
        "translation": translation_agg,
        "primary_model": primary_model,
        "summary": summary,
    }
    finish_run(run_dir, cfg, metrics, root=root, summary=summary)


if __name__ == "__main__":
    main()
