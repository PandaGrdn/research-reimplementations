# Predictive Coding

Hierarchical predictive coding on MNIST: each layer predicts the one below it, activities `r` settle on the prediction error, then the generative weights update.

Two notebooks:

| Notebook | Model |
|----------|--------|
| `predictive_coding_simply.ipynb` | Dense `U @ r` with a patch-seeded dictionary |
| `convolutional_predictive_coding.ipynb` | Tied transpose-conv (`deconv` predicts, `F.conv2d` sends error back) |

Data lives in `data/MNIST/` (60k train / 10k test).

## How it works

**Inference (inner loop).** Error at the bottom is `e0 = I − predict(r0)`. Higher layers use `ei = r_{i−1} − predict(ri)`. Activities update with a Cauchy sparsity prior and a top-down term from `e_{i+1}`.

**Learning (outer loop).** After `r` settles, generative weights move to shrink those errors (explicit `U` updates in the dense model; `loss.backward()` on deconv weights in the conv model). Columns / kernels are L2-normalized.

**Eval.** After inferring `r` on an image (no weight steps):

- **Layer 0 MSE** — reconstruct from `r[0]` (did inference fit the pixels?)
- **Layer 1 MSE** — decode only from `r[-1]` down (does the abstract code carry the image?)

Train on mean-centered digits. Persistent `r` per training image; validation uses a held-out image and a fresh `r`.

### Dense model

- `I` is a plain `28×28` flatten (`784×1`).
- Layer 0 dictionary: one `U[0]` of shape `(784, k0)`. Columns are seeded from real **7×7** tiles on a **4×4** grid (`num_patches=16`), ~32 atoms per tile.
- Dead / cloned columns can be re-seeded (`refresh_U`).
- Default run: 2 layers, 100 images, 100 epochs, linear `f`.

### Conv model

- `I` stays a feature map `(1, 1, 28, 28)`.
- Each layer is a `ConvTranspose2d` (kernel 4, stride 2, padding 1): **1 → 32** at `14×14`, then **32 → 64** at `7×7`.
- `Δr` uses `F.conv2d(e, deconv.weight, …)` — the transpose of the same kernel, not a second conv module.
- Default run: 2 layers, 100 images, 20 epochs.

## Metrics

Numbers from the saved notebook runs (mean-centered MNIST, 100 train images, held-out image 100).

**Dense (`predictive_coding_simply.ipynb`)**

| | Layer 0 MSE | Layer 1 (top-down) MSE |
|--|-------------|------------------------|
| Epoch 0 (train image 0) | 0.00042 | 0.092 |
| Epoch 99 | 0.0013 | 0.044 |
| Validation | **0.027** | **0.059** |

Bottom layer overfits quickly; the hierarchy slowly improves top-down recon (~2× better L1 MSE over training). Held-out top-down is still weak — the abstract code does not fully generalize.

**Conv (`convolutional_predictive_coding.ipynb`)**

| | Layer 0 MSE | Layer 1 (top-down) MSE |
|--|-------------|------------------------|
| Epoch 0 | 0.00098 | 0.077 |
| Epoch 19 | 0.00002 | 0.00085 |
| Validation | **0.00081** | **0.041** |

Compared with the dense model on the same 100-image setup:

- Train top-down MSE: **0.044 → 0.00085** (~50×)
- Val Layer 0 MSE: **0.027 → 0.00081** (~33×)
- Val top-down MSE: **0.059 → 0.041** (~1.4×)

Conv shares local filters, so it fits digits much more tightly. The remaining gap is still at the **highest layer on unseen images**.

## What actually improved reconstruction (dense path)

Earlier bugs produced a mean-digit blob or scrambled tiles. Fixes that mattered:

1. Hierarchical errors (`ei` predicts the layer below, not `I` at every layer).
2. Persistent per-image `r` instead of a new random code every step.
3. Patch-seeded `U` (and optional dead/clone refresh) so dictionary atoms start local and distinct.
4. Evaluate top-down from `r[-1]` as well as `U[0] @ r[0]`.
5. Linear decode + mean-centering (sigmoid sat at 0.5 / a gray mean).
6. Shared weights across the dataset, not a new `U` per image.

## Run

Open either notebook and run all cells; MNIST downloads into `./data` if needed. Need to set up an environment.
