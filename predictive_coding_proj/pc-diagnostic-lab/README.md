# Predictive-coding diagnostic lab

A codebase whose only job is to **reproduce, at publication quality, every
measurement in the diagnostic chain** behind:

> *Why hierarchical predictive coding with iterative sparse inference fails at
> autoregressive video rollout.*

This is **not** a solutions lab. The solutions testbed lives in
`../latent-rollout-lab/`. Every experiment here is evidence for one link in
the argument. The output of the whole repo is a set of `metrics.json` files
plus camera-ready figures, one per claim, plus an ablation ladder that
isolates one variable at a time.

Port of `past_attempts/convolutional_predictive_coding.ipynb` (dictionary) and
`past_attempts/temporal_predictive_coding.ipynb` (settle, ConvGRU, rollout,
diagnostics). Inference math is copied from those notebooks; flags parameterize
what the notebooks hardcoded.

A review of the first version of this lab found that several of its claims
did not actually isolate the variable they were supposed to (see "What changed
and why" below). This version reframes C4, C5, C6, C7, C8 around that review
and adds C0 (an isolation test) and C9 (a predictability metric) plus an
ablation ladder that changes exactly one thing per rung. **Every number
committed to this repo so far is a `--smoke` run** (tiny data, 1 seed) —
treat every headline number below as "what to expect," not as a reported
result. `src/experiment.py` warns loudly if you run without `--smoke` below
the reporting floor (see "Scale" below); do not quote those runs either.

---

## The argument (C0 → C9)

Each claim is one script, one results directory, one figure. A reader can
dispute any claim by rerunning exactly that script.

### C0 — the isolation test

At the ground-truth → self-prediction handoff, settle the true frame and
settle the synthetic (model-substituted) frame separately, to a convergence
tolerance rather than a fixed iteration count, and compare energy / iterations
/ cross-cosine.

```bash
python experiments/c0_isolation_test.py
```

![fig_c0](figures/fig_c0.png)

Headline number: `cos(true, synthetic)` at the handoff (and one step later).
Low cosine with comparable convergence behavior on both means the two settle
*targets* are simply different objects, not that settle is failing to
converge on one of them.

### C1 — the phenomenon

Long-horizon rollout collapses immediately at the ground-truth → self-prediction
handoff — now measured against copy-last-frame and mean-frame floors, with a
collapse-onset diagnostic (is the gap immediate, or does it compound?).

```bash
python experiments/c1_rollout_collapse.py
```

![fig_c1](figures/fig_c1.png)

MSE vs frame, `split_point` marked, copy-last / mean-frame floors overlaid at
the headline split. Headline number: mean MSE from the handoff through the
end of the sequence, read against those floors — the claim is only interesting
if the model's long-horizon MSE is at or above the copy-last floor.

### C2 — standard mitigations do not fix it

Grid over scheduled sampling, k-step rollout loss, latent noise, bounded
residual delta, the split-point bookkeeping fix (`r_prev1 ← r_curr` vs
`r_prev1 ← r_pred`), all combined, and now two temporal-prior-weight cells
(`temporal_prior_off`, `temporal_prior_strong` — see "What changed" below),
each measured against the same floors as C1.

```bash
python experiments/c2_mitigation_grid.py
```

![fig_c2](figures/fig_c2.png)

Bar chart of long-horizon MSE with copy-last / mean-frame floor lines. The
point of the figure is that every bar sits on the baseline, at or above the
floor.

### C3 — the net learns copy-last / near-identity

`motion_gap = MSE(pred, true_t) − MSE(pred, true_{t−1})` and
`delta_ratio = ‖δ‖ / ‖r_t − r_{t−1}‖`, logged every epoch.

```bash
python experiments/c3_copy_detection.py
```

![fig_c3](figures/fig_c3.png)

Positive `motion_gap` after the preds look like digits is a COPY verdict.
`delta_ratio → 0` is the identity plateau.

### C4 — the settle "noise floor" is a null-space artifact, not the pipeline's noise

`settle_grounded` starts from `torch.randn(...) * init_noise`. Two cold settles
of the same frame differ by a distance that scales ~linearly with `init_noise`
(slope ≈1 in log-log) while the PC energy converges toward 0 — the signature of
a component of the random init that lives in the **null space of an 8×
overcomplete dictionary** (layer-0 code is 32768-dim vs a 4096-pixel frame),
never driven out by gradient descent on the energy, rather than a settle
process that hasn't converged. The real train/eval pipeline never draws this
random init at inference time: it uses the fixed checkpointed `r_init` and
warm-starts every subsequent frame, so this quantity is deterministic
(`init_noise=0` gives `cos=1.000` exactly) and this "noise" never enters the
prediction target C6 measures against.

```bash
python experiments/c4_settle_determinism.py
```

![fig_c4](figures/fig_c4.png)

Left panel: distance vs `init_noise` (log-log slope ≈1 — read this off your own
run, don't trust a specific number below the reporting floor). Right panel:
cosine and energy vs iterations at the headline `init_noise`. The smoke run
here is 1 seed / 2 iterations and is not informative on its own — rerun at
the floor (≥5 seeds, iters up to 2000) before citing a slope or an asymptote.

### C5 — code smoothness against the pixel-space reference

Consecutive-frame code cosine/distance looks bad in isolation, but Moving
MNIST pixels are **also** not much smoother than "unrelated frame" pixels
under the same metric (translation makes every frame look about as different
from a random other frame as a temporal neighbor, in raw L2 terms). This
script reports the exact same consecutive-vs-unrelated ratio for four
conditions on the **same pairs**: cold-start code, warm-start code,
warm-start with zero init noise, and the raw pixel frames (no settle at all)
— plus an iteration sweep of the warm-start condition.

```bash
python experiments/c5_latent_smoothness.py
```

![fig_c5](figures/fig_c5.png)

Left panel: consec/unrelated ratio, four conditions side by side, pixel bar as
the reference the code bars must be compared against — a code ratio close to
the pixel ratio means "no worse than a space we know is fully predictable by
construction," which is the honest reading (not "smooth"). Right panel: warm
consecutive cosine vs settle iterations, pixel cosine as a flat reference
line. Headline number: consec/unrelated ratio, warm-start code vs pixel.

### C6 — how much of the prediction target is settle noise, honestly measured

Three protocols for the same question, from most to least confounded:

- `cold_independent` — the old measurement: two **cold** independent settles
  give the noise estimate, but the target norm comes from a **warm**-started
  trajectory. Mismatched protocols, kept only for comparison ("protocol
  mismatch").
- `warm_independent_init` — the honest version: run the *whole* warm-started
  trajectory twice from two different random `r_0`, so noise and target both
  come from the same protocol.
- `pipeline` — identical to the above except both trajectories start from the
  checkpointed **fixed** `r_init`, exactly as training/eval does. Settle is
  deterministic given the same warm-start chain, so `settle_abs` should be
  exactly 0 here: there is no "noise" in the object the real pipeline ever
  measures against.

`noise_floor_decomposition` (`src/metrics.py`) is **not clamped** —
`noise_over_target` can exceed 1, which would falsify "noise is a small share
of target" outright instead of hiding it. `model_consistent` reports whether
the estimated noise energy is even ≤ the target energy.

```bash
python experiments/c6_noise_floor.py
```

![fig_c6](figures/fig_c6.png)

Stacked/side-by-side bars, one per protocol: noise energy vs remaining target
energy (or an "inconsistent" flag if noise exceeds target). Headline number:
`noise_over_target` for `warm_independent_init` (the honest protocol) — this
is the number to read, not `cold_independent`.

### C7 — slowness with and without a real gradient path

`train_loop` adds `lambda_slow * mean((r_curr - r_prev1)**2)` to the weight
loss. With the *detached* settle (the old, only, path), `r_curr` is a detached
leaf by the time that term is built, so it has **zero gradient** into the
dictionary or the GRU — the old "plateau vs λ_slow" was guaranteed by
construction, not evidence about slowness. This sweep crosses `lambda_slow`
with `unroll_k`: `unroll_k=0` is the old detached settle (no gradient path, by
construction — verified directly via `slowness_has_dictionary_grad`);
`unroll_k>0` leaves the last `unroll_k` settle iterations un-detached
(`settle_with_temporal_prior_unrolled`), so slowness (and the temporal-prior
pull) actually reach the dictionary and GRU. Also reports dictionary drift
(relative Frobenius distance vs the pretrained checkpoint) and a `no_pull` vs
`with_pull` split — did the dictionary itself change shape, vs what the
train/eval pipeline actually sees at inference.

```bash
python experiments/c7_slowness_sweep.py
```

![fig_c7](figures/fig_c7.png)

Left: consecutive cosine vs λ_slow, one line per `unroll_k` (no_pull solid,
with_pull dashed). Right: dictionary drift vs λ_slow, one line per `unroll_k`.
The claim to check: does giving slowness a real gradient path (`unroll_k>0`)
change the plateau at all, and does the dictionary actually drift when it
does.

### C8 — a one-variable ladder: the inference procedure, holding everything else fixed

Every arm uses the **same** frozen pretrained dictionary, the **same**
`train_temporal_pc` training loop, and the **same** evaluators
(`validate_hierarchical`, `eval_long_rollouts`) via their `encoder=` hook.
Only how the per-frame code is produced changes:

- `iterative` — current settle (`encoder=None`).
- `amortized_init_settle` — encoder output as the settle init, then a few
  plain settle steps on the same frozen dictionary.
- `amortized` — encoder only, no settle.

This isolates the inference procedure from the five things the old C8 changed
at once (amortized vs iterative, layer count, dense vs sparse, a no-op
slowness gradient, a different temporal loop, teacher-forced vs closed-loop
MSE). Every arm is also scored on the same metrics as everywhere else: floors,
recon MSE / settle energy, latent smoothness against the same pixel reference
as C5, and predictability R² (reusing C9's own machinery, same models, same
train/test window counts) with the pixel-predictability reference computed
once, since it does not depend on the arm.

```bash
python experiments/c8_amortized_contrast.py
```

![fig_c8](figures/fig_c8.png)

Four panels, iterative / init+settle / amortized side by side: (1) teacher-
forced and long-horizon MSE vs copy-last floor, (2) latent smoothness violin
vs the pixel reference, (3) recon MSE / settle energy, (4) predictability R²
vs copy-last (bars) with the pixel-predictability reference as a line.
Causal read: if removing the settle noise floor (by removing settle) improves
smoothness and predictability *and* long-horizon MSE together, that is
evidence for the causal chain this repo is trying to establish; if it
improves smoothness/predictability but not rollout, the causal story is wrong
and something else is doing the damage.

### C9 — held-out predictability and translation consistency

C5 shows the code is about as "non-smooth" (by cosine/L2) as pixels — but that
measures geometry, not predictability, and pixels are perfectly predictable
(translation) despite scoring the same on that geometry metric. C9 asks the
sharper question directly: fit a small per-layer predictor
(context window of past codes → next residual) and evaluate its **R² against
a held-out copy-last floor** — for the code and, on the exact same windows,
for raw pixels. It also checks whether the code co-translates (rolls) with a
pure image shift at its own stride, i.e. whether the dictionary's translation
equivariance survives settle (exact-stride shifts vs aliased/non-integer-
stride shifts).

```bash
python experiments/c9_predictability.py
```

![fig_c9](figures/fig_c9.png)

Left: R² vs copy-last, code vs pixel, one bar pair per predictor (`linear`,
`conv`). R² ≤ 0 means the predictor is not even as good as predicting no
change; R² for pixels near the translation floor and R² for code near or
below 0 is the sharpest version of "the code carries less predictable
structure than the pixels it was derived from." Right: cosine between the
shifted code and the rolled original code per shift, solid bars at exact
strides, hatched bars at aliased (non-integer-stride) shifts, pixel cosine as
a reference line.

---

## The ablation ladder

`scripts/run_ladder.py` + `configs/ablation/*.yaml` run a subset of
experiments (default C1, C5, C9) once per arm, each arm changing **exactly
one thing** relative to baseline:

| arm | what changes | overlay | isolates |
|---|---|---|---|
| `baseline` | current: iterative settle, 8× overcomplete dictionary, `init_noise=0.01` | none | — |
| `zero_init` | `inference.init_noise: 0.0` | `configs/ablation/zero_init.yaml` | kills the C4 null-space "noise" — does C1/C5/C9 change at all? |
| `complete_dict` | `spatial.num_channels: 2` → layer-0 code = 4×32×32 = 4096 = pixel dim | `configs/ablation/complete_dict.yaml` | removes the null space entirely (code is no longer overcomplete) |
| `no_temporal_prior` | `inference.temporal_prior_weight: 0.0` | `configs/ablation/no_temporal_prior.yaml` | turns the (previously hard-coded, effectively-off) temporal pull fully off |
| `strong_temporal_prior` | `inference.temporal_prior_weight: 1.0` | `configs/ablation/strong_temporal_prior.yaml` | turns the temporal pull up 100× vs the notebook-parity default |
| `amortized_init_settle` / `amortized` | see C8 | (C8 arm, not a ladder overlay) | inference procedure, same dictionary/loop/evaluators |

```bash
python scripts/run_ladder.py --smoke --only c1 c5 c9
python scripts/make_figures.py     # also renders fig_ladder
```

Each arm's run is tagged `arm: <name>` in its `config.yaml`; the ladder finds
the newest run of each requested experiment whose config records that arm (not
just "latest", since arms interleave), pulls out `c1.long_mse` /
`c1.copy_last_floor`, `c5.warm_frac_rel` / `c5.pixel_frac_rel`, and
`c9.code_r2` / `c9.pixel_r2`, prints a markdown table, and writes
`results/ladder/<stamp>/metrics.json` via the same `finish_run()` every other
experiment uses.

![fig_ladder](figures/fig_ladder.png)

---

## What changed and why

A review of the first version of this lab found it did not actually isolate
"hierarchical PC with iterative settle fails at rollout" from several
confounds. Kept here, briefly and honestly, because the numbers this repo
produces are only as trustworthy as this list:

1. **Nothing had been run at the reporting floor.** Every committed result
   was `--smoke`. Still true of this version — see the warning at the top.
2. **C4's "noise floor" was a measurement artifact.** The distance between two
   cold settles of the same frame scales with `init_noise` and is the null
   space of an 8× overcomplete dictionary preserving the random init — not
   noise the real (fixed-`r_init`, warm-started, deterministic) pipeline ever
   sees. Fixed by naming it correctly and adding the `zero_init` /
   `complete_dict` ladder rungs that make it disappear by construction.
3. **C5's non-smoothness had no reference.** Pixel-space consecutive cosine on
   Moving MNIST is about as bad as the code's, under the same metric, while
   being perfectly predictable (translation). Fixed by adding the pixel
   condition to every C5 run and adding C9 (predictability, not geometry).
4. **C6's formula mixed protocols and clamped the falsifying case.** It used a
   cold-start noise estimate against a warm-start target norm, and clamped
   `noise_share` to 1.0, which would have hidden a result showing the model
   is *not* internally consistent. Fixed with three explicit protocols
   (`cold_independent` kept only as the old measurement, `warm_independent_init`
   as the honest one, `pipeline` as the deterministic real case) and an
   unclamped `noise_over_target` plus a `model_consistent` flag.
5. **C7 tested a no-op.** The slowness term sat on a detached leaf, so it had
   *zero* gradient into the dictionary or GRU by construction — any plateau
   was guaranteed, not evidence. Fixed by adding `slow_unroll_k`
   (`settle_with_temporal_prior_unrolled` leaves the last K settle steps
   un-detached) and sweeping λ_slow × unroll_k, with
   `slowness_has_dictionary_grad` verifying the mechanism directly.
6. **C8 changed five things at once** (amortized vs iterative, 1 vs 2 layers,
   dense ReLU vs sparse, working slowness vs none, a different temporal loop)
   and compared teacher-forced MSE to closed-loop MSE. Fixed by rebuilding C8
   as a same-dictionary / same-loop / same-evaluator ladder where only the
   `encoder=` hook changes, with predictability added as a fourth metric.
7. **C1/C2 had no floors.** A model that "collapses" to something *below* a
   trivial copy-last or mean-frame baseline is a different (weaker) claim than
   one that collapses to something at or above it. Fixed by adding both floors
   everywhere `validate_hierarchical(_long)` is used, and a
   collapse-onset diagnostic to C1 (is the gap immediate or does it compound?).
   The temporal-prior weight inside settle was also hard-coded to 1/100 of the
   bottom-up term (effectively off); it is now the config knob
   `inference.temporal_prior_weight`, with two C2 cells (`temporal_prior_off`,
   `temporal_prior_strong`) and two ladder rungs exercising it.
8. **Seeds were weak.** `load_splits` ignored `seed` (every seed saw the same
   sequences) and `dictionary_key` hard-coded pretrain seed 0 (every seed
   shared one dictionary) — only the GRU init actually varied across seeds.
   Fixed: `load_splits(..., seed)` selects sequences via a seeded permutation
   and records `indices_hash`; `dictionary_key(cfg, seed=...)` includes the
   seed, so each seed gets its own pretrained dictionary. This is an
   intentional departure from the original notebook, called out again under
   "Notebook parity" below.
9. **The outline's isolation test did not exist.** "Settle the true frame vs
   the synthetic frame at the handoff, compare convergence" was described but
   never scripted. That is now C0.

## Layout

```
configs/            base + spatial_pc + temporal_pc + smoke + ablation/*.yaml
src/
  data.py            MovingMNIST, fixed splits, seeded permutation + indices_hash
  spatial_pc.py       deconv dictionary + pretrain, seeded dictionary_key
  inference.py        settle_grounded + settle_info (energy/convergence) + C4/C5 diagnostics
  temporal.py         ConvGRUCell, TemporalConvRNN, train_video_sequence, slow_unroll_k
  rollout.py           teacher-forced + long rollout, copy-last/mean-frame floors, encoder= hook
  metrics.py           pair stats, motion_gap, delta_ratio, unclamped noise-floor math
  amortized.py         C8 encoder arm (PCEncoder onto the frozen dictionary)
  predictability.py    C9 predictability R² + translation consistency (also used by C8)
  experiment.py        shared CLI/config/training-loop plumbing (train_temporal_pc, ensure_dictionary, ...)
experiments/         one script per claim (c0 .. c9)
scripts/
  run_all.py           c0 → c9 → make_figures
  run_ladder.py         one-variable-at-a-time ablation ladder
  make_figures.py       every fig_cN + fig_ladder, from metrics.json only
results/<exp>/<timestamp>/metrics.json   (+ "latest" symlink)
results/ladder/<timestamp>/metrics.json
figures/fig_c0 … fig_c9, fig_ladder
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

One claim under an ablation overlay:

```bash
python experiments/c1_rollout_collapse.py --overlay configs/ablation/complete_dict.yaml
```

Everything, then figures:

```bash
python scripts/run_all.py
```

The ablation ladder:

```bash
python scripts/run_ladder.py --only c1 c5 c9
python scripts/make_figures.py
```

CPU sanity (tiny data, should finish in well under a minute):

```bash
python scripts/run_all.py --smoke
python scripts/run_ladder.py --smoke --only c1 c5 c9
python tests/test_smoke.py
```

Figures are generated **only** by `scripts/make_figures.py` from `metrics.json`.
No hand-made plots. The blog post is fully regenerable.

Every run directory stores resolved YAML, git hash, and `pip freeze`.

## Scale (refuse to quote below this)

| | Smoke | Reporting floor |
|---|---|---|
| train sequences | 1–2 | ≥ 1000 |
| held-out eval | 1 | ≥ 100 |
| pair-stat pairs | handful | ≥ 200 |
| predictability train sequences (C9/C8) | 1 | ≥ 24 |
| predictability test windows (C9/C8) | handful | ≥ 200 |
| seeds (C0/C1/C2/C3/C7/C8, `n_seeds`) | 1 | 3 |
| seeds (C4/C5/C6/C9, `n_seeds_inference`) | 1 | 5 |

`src/experiment.py` prints a warning if you are below the floor without
`--smoke`. **Every number in this repo right now is a smoke number** — none
of the above has been run at the reporting floor. Treat every headline number
in the "argument" section as an expectation to verify, not a result.

## Notebook parity

Temporal training matches `temporal_predictive_coding.ipynb` `main()`:

- `lr_r=0.02`, 200 inner iters, `lambda_slow=3`, `delta_loss_weight=100`
- bounded residual (`tanh` × `delta_scale=1`)
- annealed `ss_p` 0→0.4, `r_noise` 0→0.05, `rollout_k=4`
- `inference.temporal_prior_weight` defaults to `0.01`, which reproduces the
  notebook's hard-coded `1/(sigma_2*100)` pull exactly (algebraically
  identical for any `sigma_2`) — see
  `test_temporal_prior_weight_default_matches_old_hardcode`. C2 and the ladder
  now expose this as a knob (`temporal_prior_off` / `temporal_prior_strong`
  cells, `no_temporal_prior` / `strong_temporal_prior` ladder rungs) instead of
  leaving it hard-coded.

The lab **pins a frozen spatially-pretrained dictionary** for C0–C6, C8, C9
(cleaner isolation of the temporal failure). C7 unfreezes it on purpose, to
test whether slowness can reshape the codebook once it has a real gradient
path. The original notebook `main()` jointly trained an unfrozen dictionary
from scratch; that is the one intentional protocol difference vs. the
diagnostic cells, which already froze after spatial pretrain.

`validate_hierarchical_long` is exposed with both bookkeeping branches:

- `split_fix=False` (original): `r_prev1 ← r_pred`
- `split_fix=True` (notebook cell 18): `r_prev1 ← r_curr` settled from `I_obs`

C2 treats that difference as a mitigation cell.

Two more intentional departures from the notebooks, added by this rework and
worth calling out explicitly since they change what "seed" means: per-seed
dictionaries (`dictionary_key(cfg, seed=...)` — the notebooks pretrained one
dictionary and reused it across all runs) and seeded data splits
(`load_splits(..., seed)` — the notebooks did not vary which sequences a run
saw across seeds). Both make "3 (or 5) seeds" a real replication rather than 3
GRU inits sharing one dictionary and one dataset draw.

## Adding nothing

Do not add new mitigations or architectures here. If a number is wrong, fix the
measurement. If you want to try a fix, that belongs in `latent-rollout-lab`.
