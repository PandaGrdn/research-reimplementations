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

## The argument (C0 → C13)

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
Low cosine with comparable convergence on both means the two settle *targets*
are different objects, not that settle is failing to converge on one of them.

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

C5 measures geometry (consecutive vs unrelated cosine/L2), not predictability.
Pixels on Moving MNIST score about as "non-smooth" as the codes while remaining
perfectly predictable (translation). C9 fits a small per-layer predictor
(context window of past codes → next residual) and evaluates **R² against a
held-out copy-last floor** — for the code and, on the exact same windows, for
raw pixels. It also checks whether the code co-translates (rolls) with a pure
image shift at its own stride.

```bash
python experiments/c9_predictability.py
```

![fig_c9](figures/fig_c9.png)

Left: R² vs copy-last, code vs pixel, one bar pair per predictor (`linear`,
`conv`). R² ≤ 0 means the predictor is not even as good as predicting no
change. Right: cosine between the shifted code and the rolled original code
per shift; pixel cosine as a reference line.

### C10 — look at the pixels

Saves the trained temporal net and writes inspection grids on held-out
sequences: ground truth / teacher-forced / closed-loop AR / copy-last, so a
human can see whether the C1 numbers are hiding a blur. Reuse a checkpoint
instead of retraining with `--ckpt`.

```bash
python experiments/c10_rollout_gallery.py
```

![fig_c10](figures/fig_c10.png)

One grid per sequence under `results/c10_rollout_gallery/<stamp>/gallery/`.
Headline is seq 0 at the C1 split. If AR looks like a frozen last context
frame, that is C3 in pixels.

### C11 — offline ConvGRU ceiling on cached eval-protocol codes

C1–C9 train the GRU online against a moving settle. C11 freezes the
dictionary, caches the eval-protocol codes once, and trains two offline
ConvGRUs on those codes: **independent** (`TemporalConvRNN`, one cell +
readout per layer) and **coupled** (`CoupledTopDownRNN`, one cell on the top
layer only; the bottom layer is generated top-down through the dictionary, so
inter-layer error `e1` is exactly 0). Two predictability probes sit next to
the C9 headline: `probe_matched` (C9's protocol) and `probe_full` (minibatch
Adam over all train sequences). Also reports teacher-forced `e1` for both
models and for the true settled codes.

```bash
python experiments/c11_offline_gru.py --seeds 0 1 2
```

![fig_c11](figures/fig_c11.png)

Writes `artifacts/offline_gru_seed{k}.pt` and
`artifacts/offline_gru_coupled_seed{k}.pt` — C12 and C13 load these. Headline:
best test R² independent vs coupled, vs both probes, plus TF `e1`. If the
probes disagree by >0.1, do not quote either alone.

### C12 — per-term energy under closed-loop fade

Does PC energy rise because the rollout is failing, or because energy is
monotone in contrast (a faded digit is a cheaper reconstruction)? Three
measurements per seed: (1) **contrast sweep** — settle from zero on
`scale * I` with no rollout (the control); (2) **independent** closed-loop
energy-fade using the C11 checkpoint; (3) **coupled** closed-loop energy-fade
(`e1 ≡ 0` by construction). An earlier ~250 energy spike was a DC-centering
artifact, not a reconstruction failure.

```bash
python experiments/c12_energy_fade.py
```

![fig_c12](figures/fig_c12.png)

Read every rollout energy curve against the contrast-sweep control. Headline:
does blank-frame energy sit at the minimum, and after the handoff does `e1`
(pred/true) grow, or does normalized energy just track fade? Coupled `e1`
staying at 0 is the reference for how much of the independent `e1` growth is
a real coupling failure.

### C13 — consistency-only test-time settle (the open question)

C12 leaves open whether *doing something* with the inter-layer residual at
test time can stop closed-loop fade. C13 is that experiment: after the
handoff, settle the rolled-out code on the **inter-layer consistency term
alone** — no image term. The decoded frame is a deterministic function of the
code, so settling on it is information-free; only hierarchy + priors can
help. Every arm shares the same teacher-forced prefix, then differs only
after `split_point`:

| arm | what is fed back as `r_prev` |
|---|---|
| `none` | pure latent `r_pred` (C12 baseline) |
| `consistency_top_down` | settle `r0` with `r1` fixed |
| `consistency_bottom_up` | settle `r1` with `r0` fixed |
| `consistency_joint` | settle both layers |
| `image_settle` | full settle on the model's **own** decode (info-free reference) |
| `oracle_image` | full settle on the **true** frame (upper reference) |

Both model kinds from C11. On the coupled model `e1` is 0 by construction, so
the consistency arms are asserted no-ops (`e1_before < 1e-6`) — that
assertion **is** the control.

Default `c13.n_sequences` is 32, which is **below** the ≥100 held-out
reporting floor even when `data.n_test: 100`. Set `c13.n_sequences: 100`
before quoting.

```bash
python experiments/c13_consistency_rollout.py --seeds 0 1 2
```

![fig_c13](figures/fig_c13.png)

Three panels: post-split pixel MSE per arm (independent vs coupled, copy-last
floor), latent cosine drift vs step, fade (`mass_pred`) vs step. Headline:
does any consistency arm beat `none` on post-split MSE, and by how much
relative to `oracle_image`? If consistency cannot, and only the oracle
(real pixels) can, the hierarchy has no unused consistency signal that
test-time settle can recover — that is the resolution of the blog's open
question.

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
configs/            base + spatial_pc + temporal_pc + smoke + reporting + ablation/*.yaml
src/
  data.py              MovingMNIST, fixed splits, seeded permutation + indices_hash
  spatial_pc.py        deconv dictionary + pretrain, seeded dictionary_key
  inference.py         settle_grounded + settle_info + C4/C5 diagnostics
  temporal.py          ConvGRUCell, TemporalConvRNN, CoupledTopDownRNN, slow_unroll_k
  rollout.py           teacher-forced + long rollout, copy-last/mean-frame floors
  metrics.py           pair stats, motion_gap, delta_ratio, unclamped noise-floor math
  amortized.py         C8 encoder arm
  predictability.py    C9 R² + translation consistency
  offline_gru.py       C11 cached-code GRUs
  energy_fade.py       C12 contrast sweep + per-term energy
  consistency_settle.py C13 inter-layer-only settle
  experiment.py        shared CLI/config/training-loop plumbing
experiments/           one script per claim (c0 … c13)
scripts/
  run_all.py           c0 → c13 → make_figures (C2/C7 via run_parallel)
  run_parallel.py      C1/C2/C3/C7/C8 fan-out + merge
  farmshare_submit.sh  Slurm arrays for the same jobs
  run_ladder.py        one-variable ablation ladder
  make_figures.py      every fig_cN + fig_ladder, from metrics.json only
colab_run_lab.ipynb    Colab GPU driver (clones repo, shells out to .py)
results/<exp>/<timestamp>/metrics.json   (+ "latest" symlink)
figures/fig_c0 … fig_c13, fig_ladder
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

Colab GPU (no need to convert scripts to notebooks): open `colab_run_lab.ipynb`,
**Select Kernel → Colab → GPU**, run all cells. C2 / C7 are launched with
`scripts/run_parallel.py` (one process per cell×seed). On one GPU keep
`--workers 2`; on CPU use 4–8. `run_all.py` routes C2/C7 through that runner.

```bash
python scripts/run_parallel.py c2 --workers 8 --device cpu --seeds 0 1 2 --overlay configs/reporting.yaml
python scripts/run_parallel.py c7 --workers 8 --device cpu --seeds 0 1 2
```

`configs/reporting.yaml` raises `eval.n_rollout_sequences` from 32 to the
100-sequence floor; without it C1/C2/C3/C8 rollouts are scored on 32 sequences.

Reporting-scale C2 is 9 cells × 3 seeds = 27 jobs; C7 is 2 × 6 × 3 = 36 jobs.
One job is 1000 sequences × 20 epochs = 20,000 sequence passes of a
single-threaded Python settle loop, so wall clock follows the number of fast
cores, not the GPU. Measured on an Apple M5 (4 performance + 6 efficiency
cores, on battery):

| setup | sequences / s |
|---|---|
| 1 MPS process | 0.82 |
| 4 CPU workers | 2.18 |
| 8 CPU workers | 2.78 |
| 8 CPU workers + 1 MPS | 3.07 |

That projects to about 54 h for C2 and 89 h for C7 on that laptop. With one
core per job (a cluster node, or a Slurm array) the grid takes roughly one job
length: about 9 h on an M5 performance core, longer on slower cluster cores.

Each job logs to `results/<exp>/<stamp>/shards/<job>/log.txt`. If jobs fail or
the machine sleeps, rerun the same command with `--resume`; finished shards are
skipped. The merge refuses an incomplete grid unless `--allow-partial`.

```bash
python scripts/run_parallel.py c2 --resume results/c2_mitigation_grid/STAMP --workers 8 --device cpu --seeds 0 1 2 --overlay configs/reporting.yaml
```

The same runner fans C1, C3 and C8 out one process per seed, and its merge is
exactly what a sequential multi-seed run writes (`tests/test_parallel.py`
compares every key):

```bash
python scripts/run_parallel.py c1 --workers 3 --device cpu --seeds 0 1 2 --overlay configs/reporting.yaml
python scripts/run_parallel.py c8 --workers 3 --device cpu --seeds 0 1 2 --overlay configs/reporting.yaml
```

C3 needs no training once C1 has run under the same config. It trains the same
model with the same seed and only reads the per-epoch `motion_gap` /
`delta_ratio` that C1 already stores. It refuses a C1 run whose training config
differs:

```bash
python experiments/c3_copy_detection.py --from-c1 results/c1_rollout_collapse/latest --overlay configs/reporting.yaml
```

### Slurm (Stanford FarmShare)

`scripts/farmshare_submit.sh` submits, per experiment, one warmup job, one array
with one single-core task per job, and one merge job. Every task of an
experiment writes into `results/<exp>/farmshare`, so resubmitting the same
command only reruns what is missing.

```bash
ssh SUNETID@login.farmshare.stanford.edu
git clone https://github.com/PandaGrdn/research-reimplementations.git
cd research-reimplementations/predictive_coding_proj/pc-diagnostic-lab
bash scripts/farmshare_submit.sh setup             # venv, CPU torch, dataset, tests
bash scripts/farmshare_submit.sh pilot             # one 1/10-scale job, to time a core
bash scripts/farmshare_submit.sh run c1 c3 c2 c8   # 33 single-core tasks; c3 is read out of c1
bash scripts/farmshare_submit.sh run c7            # 36 more
squeue -u $USER
```

Then, from the laptop:

```bash
rsync -av --exclude 'shards/*/seed*' SUNETID@login.farmshare.stanford.edu:research-reimplementations/predictive_coding_proj/pc-diagnostic-lab/results/ results/
python scripts/make_figures.py
```

There is no mid-job checkpoint, so the time limit must exceed one job. A full
job is about 10x the pilot's elapsed time. `DRY_RUN=1` prints the `sbatch`
commands without submitting. The script was exercised end to end against a
stand-in `sbatch`, not on FarmShare itself; partition, time and module names are
environment variables at the top of the script.

By hand, the three steps are:


```bash
RUN=results/c2_mitigation_grid/farmshare
python scripts/run_parallel.py c2 --warmup-only --device cpu
# in the array script (#SBATCH --array=0-26 --cpus-per-task=1):
python scripts/run_parallel.py c2 --resume $RUN --shard $SLURM_ARRAY_TASK_ID/27 --workers 1 --device cpu --overlay configs/reporting.yaml
# when the array is done:
python scripts/run_parallel.py c2 --merge-only $RUN --overlay configs/reporting.yaml
```

A merge must be given the same `--cells` / `--seeds` / `--smoke` / `--overlay`
flags as the run, since they define which shards are expected.

CPU sanity for the parallel path:

```bash
python tests/test_parallel.py
python scripts/run_parallel.py c2 --smoke --cells baseline --seeds 0 --workers 2 --no-latest
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
python tests/test_parallel.py
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
| C11/C12/C13 eval sequences | 1 | ≥ 100 (`configs/reporting.yaml`; base C13 is 32) |
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

The lab **pins a frozen spatially-pretrained dictionary** for C0–C6 and C8–C13
(cleaner isolation of the temporal failure). C7 unfreezes it on purpose, to
test whether slowness can reshape the codebook once it has a real gradient
path. C11–C13 further freeze the *codes*: they train or load offline GRUs on
cached eval-protocol settles rather than the notebook's joint online loop. The
original notebook `main()` jointly trained an unfrozen dictionary from scratch;
that is the one intentional protocol difference vs. the diagnostic cells, which
already froze after spatial pretrain.

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
