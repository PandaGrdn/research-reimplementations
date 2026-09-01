# Latent rollout lab

Modular testbed for **autoregressive rollout degradation** in latent world models. The thing under test is a **pluggable training strategy**; data, frozen autoencoder, ConvGRU, and evaluation stay fixed. Every experiment is a YAML config, not a code edit.

This is a port of `encoder_based_temporal.ipynb` (curriculum + burn-in + residual head). Toy notebook scale is for smoke tests; paper claims need the scale protocol below.

## Layout

```
configs/           base + strategy + model YAMLs
src/data.py
src/models/        autoencoder, convgru, registry
src/strategies/    THE research surface
src/train.py       strategy-agnostic trainer
src/evaluate.py    fixed metric suite
scripts/run_experiment.py
scripts/run_baselines.py
scripts/make_report.py
```

## Setup

```bash
cd latent-rollout-lab
pip install -r requirements.txt
```

Moving MNIST is loaded from `../data` (same npy as the notebooks). It downloads if missing.

## Run an experiment

```bash
python scripts/run_experiment.py configs/strategy/curriculum_rollout.yaml
```

Optional: `--seeds 0 1 2`, `--epochs 25`, `--n-train 1200`.

This will:

1. Pretrain and pin a frozen AE (`artifacts/ae.pt`) if the checkpoint is absent
2. Cache latents
3. Train the temporal model with the strategy in the YAML
4. Write `results/<name>_seed<k>_<stamp>/` with resolved config, git hash, `pip freeze`, checkpoint, metrics, rollout panels

## Add a new solution

1. Subclass `TrainingStrategy` in `src/strategies/<name>.py`
2. Implement `compute_loss(model, ae, latents, frames, epoch)` (handle burn-in via `model.init_hidden_from_context`)
3. Register the class in `src/models/registry.py` `STRATEGIES`
4. Add `configs/strategy/<name>.yaml` with `strategy.name: <name>`

Nothing else should change. Trainer, eval, and report stay frozen.

`compute_loss` must return `{"loss": tensor, "logs": {...}}`.

## Baselines (always overlay on comparison plots)

| Strategy | Config |
|---|---|
| Teacher-forcing only (1-step lower bound) | `configs/strategy/teacher_forcing.yaml` |
| Incumbent curriculum (TF warmup → unroll 1/2/4/8/12) | `configs/strategy/curriculum_rollout.yaml` |
| Scheduled sampling | `configs/strategy/scheduled_sampling.yaml` |
| Copy-last-frame / mean-frame floors | computed in `evaluate.py`, not a trained model |

```bash
python scripts/run_baselines.py --seeds 0 1 2
python scripts/make_report.py
```

`results/report/` gets a markdown table (context × horizon) and per-step AR curves.

## Experiment protocol

- **Fixed eval:** same frozen AE, same test split, 3 seeds, report mean±std.
- **Primary metric:** per-step AR MSE/SSIM at `context=2` and `context=8`. Headline number = mean over rollout steps 5–18 (long-horizon regime).
- **Secondary:** latent cosine-drift curve `cos(r_pred_t, r_true_t)` — distinguishes representation drift from decoder blur.
- **FVD:** not implemented. Needed for a real paper; heavy; add later.

## Scale (before drawing conclusions)

| | Smoke / notebook | Paper target |
|---|---|---|
| `N_TRAIN` | 32–1200 | ≥ 2000 |
| AE | 1 conv layer | ≥ 2 conv layers |
| Temporal epochs | 1–25 | ≥ 100 |
| Seeds | 1 | 3 |

Current notebook defaults (`n_train=1200`, 25 temporal epochs, 1-layer AE) are **not** enough to claim a solution to latent AR failure.

## Reproducibility

Every run directory stores:

- resolved YAML
- git hash
- `pip freeze`
- split hash (sequence indices + first/last tensor bytes)

`python tests/test_smoke.py` — 1 epoch, 32 sequences, all three strategies, eval + JSON. Should finish in a few minutes on CPU.

## Burn-in (not a strategy)

All rollouts teacher-force the GRU over the observed context, then autoregress with the model's own latents. That is eval protocol, not the independent variable. Strategies may only change **how** the model is trained to survive that rollout.
