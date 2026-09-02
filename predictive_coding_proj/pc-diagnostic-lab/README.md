# Predictive-coding diagnostic lab

A codebase whose only job is to **reproduce, at publication quality, every
measurement in the diagnostic chain** behind:

> *Why hierarchical predictive coding with iterative sparse inference fails at
> autoregressive video rollout.*

This is **not** a solutions lab. The solutions testbed lives in
`../latent-rollout-lab/`. Every experiment here is evidence for one link in the
argument. The output of the whole repo is a set of `metrics.json` files plus
camera-ready figures, one per claim.

Port of `past_attempts/convolutional_predictive_coding.ipynb` (dictionary) and
`past_attempts/temporal_predictive_coding.ipynb` (settle, ConvGRU, rollout,
diagnostics). Inference math is copied from those notebooks; flags parameterize
what the notebooks hardcoded.

---

## The argument (C1 → C8)

Each claim is one script, one results directory, one figure. A reader can
dispute any claim by rerunning exactly that script.

### C1 — the phenomenon

Long-horizon rollout collapses immediately at the ground-truth → self-prediction
handoff.

```bash
python experiments/c1_rollout_collapse.py
```

![fig_c1](figures/fig_c1.png)

MSE vs frame, `split_point` marked. Headline number: mean MSE from the handoff
through the end of the sequence.

### C2 — standard mitigations do not fix it

Grid over scheduled sampling, k-step rollout loss, latent noise, bounded
residual delta, and the split-point bookkeeping fix (`r_prev1 ← r_curr` vs
`r_prev1 ← r_pred`), each alone and all combined.

```bash
python experiments/c2_mitigation_grid.py
```

![fig_c2](figures/fig_c2.png)

Bar chart of long-horizon MSE. The point of the figure is that every bar sits
on the baseline.

### C3 — the net learns copy-last / near-identity

`motion_gap = MSE(pred, true_t) − MSE(pred, true_{t−1})` and
`delta_ratio = ‖δ‖ / ‖r_t − r_{t−1}‖`, logged every epoch.

```bash
python experiments/c3_copy_detection.py
```

![fig_c3](figures/fig_c3.png)

Positive `motion_gap` after the preds look like digits is a COPY verdict.
`delta_ratio → 0` is the identity plateau.

### C4 — iterative settle has an irreducible noise floor

Same frame, two random inits, cosine/relative distance on a grid of iterations
and `lr_r`.

```bash
python experiments/c4_settle_determinism.py
```

![fig_c4](figures/fig_c4.png)

The 0.73 → 0.93 cosine-vs-iterations curve, with the residual floor called out.
Even 400 iters does not give a unique code.

### C5 — the sparse latent trajectory is temporally non-smooth

Consecutive vs unrelated pair stats, cold and warm start, ≥200 frame pairs.

```bash
python experiments/c5_latent_smoothness.py
```

![fig_c5](figures/fig_c5.png)

Headline number: consecutive / unrelated distance ratio. Near 1 means consecutive
frames are as far apart in `r` as unrelated frames.

### C6 — the prediction target is dominated by noise

Decompose settle noise from C4 against `‖r_t − r_{t−1}‖` from C5.
Energy identity: two independent settles differ by `‖ε₁−ε₂‖`, so per-inference
`σ = ‖ε₁−ε₂‖ / √2` and the noise energy in a consecutive difference is
`2σ²`. Predictable fraction = `1 − 2σ² / ‖Δr‖²`.

```bash
python experiments/c6_noise_floor.py
```

![fig_c6](figures/fig_c6.png)

Stacked bar: noise share vs residual signal. This is the article's centerpiece
number. C6 **re-measures** settle noise and target norm itself (it does not
read C4/C5 files), so it can be disputed in isolation.

### C7 — slowness cannot fix it in-place

`λ_slow ∈ {0, 0.1, 0.5, 1, 2, 5}` → post-train consecutive cosine.

```bash
python experiments/c7_slowness_sweep.py
```

![fig_c7](figures/fig_c7.png)

Smoothness vs `λ_slow` plateaus. There is **no gradient path from the slowness
term to the dictionary**: after settle, `r_curr` is a detached leaf, so
`λ_slow ‖r_t − r_{t−1}‖²` in the weight loss does not reshape the codebook.
The only effect is a pull during inference, which is an architectural ceiling.

### C8 — swapping iterative inference for an amortized encoder

Minimal single-conv autoencoder with slowness in the weight loss. Same-frame
cosine is 1.0 by construction. Consecutive/unrelated stats and short rollout
MSE are compared to the iterative PC arm.

```bash
python experiments/c8_amortized_contrast.py
```

![fig_c8](figures/fig_c8.png)

Side-by-side C5-style violins, iterative vs amortized. Causal confirmation:
remove the noise floor, smoothness improves, rollout improves with it.

---

## Layout

```
configs/           base + spatial_pc + temporal_pc + smoke
src/
  data.py          MovingMNIST, fixed splits, seeded
  spatial_pc.py    deconv dictionary + pretrain
  inference.py     settle_grounded + C4/C5 diagnostics
  temporal.py      ConvGRUCell, TemporalConvRNN, train_video_sequence
  rollout.py       teacher-forced + long rollout (both split-fix branches)
  metrics.py       pair stats, motion_gap, delta_ratio, noise-floor math
  amortized.py     C8 encoder arm
experiments/       one script per claim
scripts/run_all.py
scripts/make_figures.py
results/<exp>/<timestamp>/metrics.json
figures/fig_c1 … fig_c8
```

## Setup

```bash
cd pc-diagnostic-lab
pip install -r requirements.txt
```

Moving MNIST is loaded from `../data` (same npy as the notebooks / solutions
lab). It downloads if missing.

## Run

One claim:

```bash
python experiments/c4_settle_determinism.py
```

Everything, then figures:

```bash
python scripts/run_all.py
```

CPU sanity (tiny data, should finish in a few minutes):

```bash
python scripts/run_all.py --smoke
python tests/test_smoke.py
```

Figures are generated **only** by `scripts/make_figures.py` from `metrics.json`.
No hand-made plots. The blog post is fully regenerable.

Every run directory stores resolved YAML, git hash, and `pip freeze`.

## Scale (refuse to quote below this)

| | Smoke | Reporting floor |
|---|---|---|
| train sequences | 2 | ≥ 1000 |
| held-out eval | 1 | ≥ 100 |
| pair-stat pairs | handful | ≥ 200 |
| seeds | 1 | 3 (5 for C4/C5) |

`src/experiment.py` prints a warning if you are below the floor without `--smoke`.

## Notebook parity

Temporal training matches `temporal_predictive_coding.ipynb` `main()`:

- `lr_r=0.02`, 200 inner iters, `lambda_slow=3`, `delta_loss_weight=100`
- bounded residual (`tanh` × `delta_scale=1`)
- annealed `ss_p` 0→0.4, `r_noise` 0→0.05, `rollout_k=4`

The lab **pins a frozen spatially-pretrained dictionary** for C1–C6 and C8
(cleaner isolation of the temporal failure). C7 unfreezes it on purpose, to
show that slowness still cannot reshape the codebook. The original notebook
`main()` jointly trained an unfrozen dictionary from scratch; that is the one
intentional protocol difference vs. the diagnostic cells, which already froze
after spatial pretrain.

`validate_heirarchical_long` is exposed with both bookkeeping branches:

- `split_fix=False` (original): `r_prev1 ← r_pred`
- `split_fix=True` (notebook cell 18): `r_prev1 ← r_curr` settled from `I_obs`

C2 treats that difference as a mitigation cell.

## Adding nothing

Do not add new mitigations or architectures here. If a number is wrong, fix the
measurement. If you want to try a fix, that belongs in `latent-rollout-lab`.
