"""Offline ConvGRU on cached eval-protocol codes, plus energy-vs-fade logging.

Eval-protocol codes = warm `settle_grounded` (no λ_slow, no temporal pull).
The GRU is the same `TemporalConvRNN` as the in-loop model; only the
optimizer and the training targets change (Adam on cached codes).
"""

import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.inference import settle_grounded, settle_info
from src.metrics import flat_r, pair_stats, stack_mean_std
from src.predictability import _build_model, _build_windows, encode_sequences, make_settle_encoder
from src.spatial_pc import dictionary_key, f_clamp
from src.utils import clone_r


def encode_eval_codes(seqs, deconvs, r_init, cfg, root=None, seed=None, split_name="eval"):
    """Warm settle_grounded over each sequence. Returns per-seq per-frame codes.

    Thin wrapper: when both `root` and `seed` are given, delegates to the
    disk-cached `cache_eval_codes`; otherwise (e.g. --smoke runs, which must
    never read or write the cache) computes directly, unchanged from before.
    """
    if root is not None and seed is not None:
        return cache_eval_codes(split_name, seqs, deconvs, r_init, cfg, root, seed)
    encoder = make_settle_encoder(deconvs, r_init, cfg, warm=True)
    return encode_sequences(seqs, encoder)


def _codes_cache_key(cfg, seed, n_sequences):
    """dictionary_key(cfg, seed) plus a short hash of everything that changes
    what settle_grounded actually produces but that dictionary_key does not
    already cover: how many sequences were encoded and
    inference.num_epochs_inner / lr_r / init_noise. Without this, a changed
    inner-loop or scale setting would silently reuse stale cached codes.
    """
    base = dictionary_key(cfg, seed=seed)
    inf = cfg.get("inference", {})
    extra = {
        "n_sequences": n_sequences,
        "num_epochs_inner": inf.get("num_epochs_inner"),
        "lr_r": inf.get("lr_r"),
        "init_noise": inf.get("init_noise", 0.01),
    }
    blob = json.dumps(extra, sort_keys=True, default=str).encode()
    extra_hash = hashlib.sha256(blob).hexdigest()[:8]
    return f"{base}_{extra_hash}"


def cache_eval_codes(split_name, seqs, deconvs, r_init, cfg, root, seed, log=print):
    """Warm settle_grounded codes for `split_name`, disk-cached at
    artifacts/codes_{split_name}_seed{seed}_{key}.pt.

    key = dictionary_key(cfg, seed) plus a hash of (n_sequences,
    inference.num_epochs_inner, lr_r, init_noise) — see `_codes_cache_key`.
    Tensors are saved on CPU and returned on the device `r_init` already
    lives on (the cfg device). C11/C12/C13 must all go through this so they
    train / evaluate / settle on identical codes. Not used under --smoke
    (callers should call `encode_eval_codes(...)` with no `root`/`seed` there).
    """
    device = r_init[0].device
    key = _codes_cache_key(cfg, seed, len(seqs))
    path = Path(root) / "artifacts" / f"codes_{split_name}_seed{seed}_{key}.pt"
    if path.exists():
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
        codes = [[[ri.to(device) for ri in frame] for frame in seq] for seq in raw]
        log(f"cache_eval_codes[{split_name}]: loaded {len(codes)} sequences from {path}")
        return codes

    codes = encode_eval_codes(seqs, deconvs, r_init, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_codes = [[[ri.detach().cpu() for ri in frame] for frame in seq] for seq in codes]
    torch.save(cpu_codes, path)
    log(f"cache_eval_codes[{split_name}]: computed {len(codes)} sequences, saved to {path}")
    return codes


def _layer_mse(pred, target):
    return sum(F.mse_loss(a, b) for a, b in zip(pred, target)) / max(len(pred), 1)


def gru_r2(model, codes):
    """Teacher-forced R² vs copy-last: 1 − ||r̂_t − r_t||² / ||r_{t−1} − r_t||²."""
    sse_pred = 0.0
    sse_copy = 0.0
    n_windows = 0
    model.eval()
    with torch.no_grad():
        for seq in codes:
            hidden = model.init_hidden(seq[0])
            for t in range(1, len(seq)):
                r_pred, hidden, _ = model(seq[t - 1], hidden)
                for a, b, c in zip(r_pred, seq[t], seq[t - 1]):
                    sse_pred += float(((a - b) ** 2).sum().item())
                    sse_copy += float(((c - b) ** 2).sum().item())
                n_windows += 1
    return {
        "r2_vs_copy_last": float(1.0 - sse_pred / (sse_copy + 1e-12)),
        "mse_pred": float(sse_pred / max(n_windows, 1)),
        "mse_copy_last": float(sse_copy / max(n_windows, 1)),
        "n_windows": n_windows,
    }


def gru_r2_per_layer(model, codes):
    """Per-layer breakdown of `gru_r2`'s teacher-forced R² vs copy-last.

    Needed for the coupled model (C11): its layer-0 prediction is generated
    top-down through the dictionary rather than fit directly, so its R² can
    (and is expected to) look very different from layer 1's.
    """
    num_layers = len(codes[0][0])
    sse_pred = [0.0] * num_layers
    sse_copy = [0.0] * num_layers
    n_windows = 0
    model.eval()
    with torch.no_grad():
        for seq in codes:
            hidden = model.init_hidden(seq[0])
            for t in range(1, len(seq)):
                r_pred, hidden, _ = model(seq[t - 1], hidden)
                for l in range(num_layers):
                    a, b, c = r_pred[l], seq[t][l], seq[t - 1][l]
                    sse_pred[l] += float(((a - b) ** 2).sum().item())
                    sse_copy[l] += float(((c - b) ** 2).sum().item())
                n_windows += 1
    per_layer = [
        {
            "layer": l,
            "r2_vs_copy_last": float(1.0 - sse_pred[l] / (sse_copy[l] + 1e-12)),
            "mse_pred": float(sse_pred[l] / max(n_windows, 1)),
            "mse_copy_last": float(sse_copy[l] / max(n_windows, 1)),
        }
        for l in range(num_layers)
    ]
    return {"per_layer": per_layer, "n_windows": n_windows}


def teacher_forced_e1(model, codes, deconvs):
    """Teacher-forced inter-layer consistency error, mean over windows of
    ||r_pred0 - f_clamp(deconv1(r_pred1))||² / 2.

    `model(seq[t-1], hidden)` is teacher-forced exactly like `gru_r2`: the
    hidden state advances autoregressively but the *input* r at every step is
    the true previous code, never the model's own previous prediction.
    """
    total = 0.0
    n_windows = 0
    model.eval()
    with torch.no_grad():
        for seq in codes:
            hidden = model.init_hidden(seq[0])
            for t in range(1, len(seq)):
                r_pred, hidden, _ = model(seq[t - 1], hidden)
                target = f_clamp(deconvs[1](r_pred[1]))
                e1 = r_pred[0] - target
                total += float((e1 ** 2).sum().item()) / 2.0
                n_windows += 1
    return total / max(n_windows, 1)


def true_codes_e1(codes, deconvs):
    """Inter-layer consistency error of the actual settled (true) codes
    themselves — no model, no teacher forcing. The reference point background
    findings call "≈0.25 for true codes" (vs ≈0.7 for GRU predictions)."""
    total = 0.0
    n_windows = 0
    with torch.no_grad():
        for seq in codes:
            for frame in seq:
                target = f_clamp(deconvs[1](frame[1]))
                e1 = frame[0] - target
                total += float((e1 ** 2).sum().item()) / 2.0
                n_windows += 1
    return total / max(n_windows, 1)


def probe_r2_minibatch(
    train_codes,
    test_codes,
    device,
    context=2,
    epochs=20,
    lr=1e-3,
    hidden=32,
    seed=0,
    model="conv",
    batch_size=64,
    log=print,
):
    """Minibatched analogue of `predictability.predictability_r2`.

    `predictability_r2` is full-batch Adam for a fixed step count; at
    ~1000 training sequences that underfits badly (see SPEC2 background: it
    reported R²=-0.04 there vs C9's 0.51 on 24 sequences). This trains the
    same per-layer conv predictor with Adam over shuffled minibatches of
    `batch_size` windows, `epochs` passes over ALL training windows, and
    reports weighted (by window count) train/test R² vs copy-last after every
    epoch plus the best test R² seen — the "does the ceiling actually
    converge at this scale" check that C11's probe_full uses.
    """
    train_in, train_tgt, num_layers = _build_windows(train_codes, context)
    test_in, test_tgt, _ = _build_windows(test_codes, context)

    nets, opts, x_trs, y_trs, x_tes, y_tes = [], [], [], [], [], []
    for l in range(num_layers):
        x_tr, y_tr = train_in[l].to(device), train_tgt[l].to(device)
        x_te, y_te = test_in[l].to(device), test_tgt[l].to(device)
        x_trs.append(x_tr)
        y_trs.append(y_tr)
        x_tes.append(x_te)
        y_tes.append(y_te)
        in_ch, out_ch = x_tr.shape[1], y_tr.shape[1]
        torch.manual_seed(seed * 1000 + l)
        net = _build_model(in_ch, out_ch, hidden, model).to(device)
        nets.append(net)
        opts.append(torch.optim.Adam(net.parameters(), lr=lr))

    n_train_windows = int(x_trs[0].shape[0]) if x_trs else 0
    n_test_windows = int(x_tes[0].shape[0]) if x_tes else 0
    history = []
    best_test_r2 = -float("inf")
    best_epoch = -1
    for epoch in range(epochs):
        perm = torch.randperm(max(n_train_windows, 1))[:n_train_windows]
        for net in nets:
            net.train()
        for start in range(0, n_train_windows, batch_size):
            idx = perm[start : start + batch_size]
            for l in range(num_layers):
                opts[l].zero_grad()
                loss = F.mse_loss(nets[l](x_trs[l][idx]), y_trs[l][idx])
                loss.backward()
                opts[l].step()

        sum_mse_tr = sum_copy_tr = 0.0
        sum_mse_te = sum_copy_te = 0.0
        n_tr = n_te = 0
        for l in range(num_layers):
            nets[l].eval()
            with torch.no_grad():
                mse_tr = float(F.mse_loss(nets[l](x_trs[l]), y_trs[l]).item())
                copy_tr = float(torch.mean(y_trs[l] ** 2).item())
                mse_te = float(F.mse_loss(nets[l](x_tes[l]), y_tes[l]).item())
                copy_te = float(torch.mean(y_tes[l] ** 2).item())
            numel_tr, numel_te = int(y_trs[l].numel()), int(y_tes[l].numel())
            sum_mse_tr += mse_tr * numel_tr
            sum_copy_tr += copy_tr * numel_tr
            n_tr += numel_tr
            sum_mse_te += mse_te * numel_te
            sum_copy_te += copy_te * numel_te
            n_te += numel_te

        train_r2 = float(1.0 - (sum_mse_tr / max(n_tr, 1)) / (sum_copy_tr / max(n_tr, 1) + 1e-12))
        test_r2 = float(1.0 - (sum_mse_te / max(n_te, 1)) / (sum_copy_te / max(n_te, 1) + 1e-12))
        history.append({"epoch": epoch, "train_r2": train_r2, "test_r2": test_r2})
        log(f"[probe_minibatch {epoch:03d}/{epochs:03d}]  train_r2={train_r2:.4f}  test_r2={test_r2:.4f}")
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_epoch = epoch

    return {
        "history": history,
        "best_test_r2": best_test_r2,
        "best_epoch": best_epoch,
        "final_test_r2": history[-1]["test_r2"] if history else float("nan"),
        "final_train_r2": history[-1]["train_r2"] if history else float("nan"),
        "n_train_windows": n_train_windows,
        "n_test_windows": n_test_windows,
        "context": context,
        "model": model,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def train_offline_gru_full(model, train_codes, test_codes, epochs, lr, device, log=print):
    """Adam on teacher-forced next-code MSE for the FULL `epochs`, no early
    stop. Tracks per-epoch test R² and keeps a deep copy of the state_dict at
    the best (highest test R²) epoch — the checkpointed model is the best one
    seen even though training always runs to the end. Separate from
    `train_offline_gru` (which early-stops on `target_r2` and is still used
    by C12) rather than changing that function's behaviour under it.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_r2 = -float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seq = 0
        for seq in train_codes:
            hidden = model.init_hidden(seq[0])
            loss = 0.0
            steps = 0
            for t in range(1, len(seq)):
                r_in = [ri.detach() for ri in seq[t - 1]]
                r_tgt = [ri.detach() for ri in seq[t]]
                r_pred, hidden, _ = model(r_in, hidden)
                loss = loss + _layer_mse(r_pred, r_tgt)
                steps += 1
            loss = loss / max(steps, 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().item())
            n_seq += 1
        train_loss = running / max(n_seq, 1)
        stats = gru_r2(model, test_codes)
        rec = {"epoch": epoch, "train_loss": train_loss, "test_r2": stats["r2_vs_copy_last"]}
        history.append(rec)
        log(
            f"[offline GRU {epoch:03d}/{epochs:03d}]  train_mse={train_loss:.6f}  "
            f"test R²={stats['r2_vs_copy_last']:.4f}"
        )
        if stats["r2_vs_copy_last"] > best_r2:
            best_r2 = stats["r2_vs_copy_last"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    return {
        "history": history,
        "best_r2": best_r2,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "final_r2": history[-1]["test_r2"] if history else float("nan"),
    }


def train_offline_gru(
    model,
    train_codes,
    test_codes,
    epochs,
    lr,
    device,
    target_r2=None,
    log=print,
):
    """Adam on teacher-forced next-code MSE. Logs test R² each epoch; optional early stop."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_seq = 0
        for seq in train_codes:
            hidden = model.init_hidden(seq[0])
            loss = 0.0
            steps = 0
            for t in range(1, len(seq)):
                r_in = [ri.detach() for ri in seq[t - 1]]
                r_tgt = [ri.detach() for ri in seq[t]]
                r_pred, hidden, _ = model(r_in, hidden)
                loss = loss + _layer_mse(r_pred, r_tgt)
                steps += 1
            loss = loss / max(steps, 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().item())
            n_seq += 1
        train_loss = running / max(n_seq, 1)
        stats = gru_r2(model, test_codes)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_r2": stats["r2_vs_copy_last"],
            "test_mse_pred": stats["mse_pred"],
            "test_mse_copy_last": stats["mse_copy_last"],
        }
        history.append(rec)
        log(
            f"[offline GRU {epoch:03d}/{epochs:03d}]  train_mse={train_loss:.6f}  "
            f"test R² vs copy-last={stats['r2_vs_copy_last']:.4f}"
        )
        if target_r2 is not None and stats["r2_vs_copy_last"] >= target_r2:
            log(f"hit target R²={target_r2:.3f} at epoch {epoch}")
            break
    return history


def _prepare_frame(frame):
    x = frame.float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x - x.mean()


def _contrast(frame):
    x = frame.detach()
    return float(x.abs().mean().item()), float(x.abs().amax().item())


def energy_fade_rollout(seq, r_init, deconvs, temporal_nn, cfg, split_point):
    """Pure-latent closed-loop after `split_point`; energy of true vs faded frames.

    Before the handoff the GRU is teacher-forced on eval-protocol codes.
    After it, r_prev ← r_pred (no re-settle). At every t≥1 we log:

      energy_true       settle_info(I_t, r_true)
      energy_faded      settle_info(Î_t, r_settled_on_Î)
      energy_pred_code  settle_info(Î_t, r_pred)   (no extra settle)
      mass/peak         of I_t and Î_t
      cos               (r_pred, r_true)
    """
    inf = cfg["inference"]
    num_layers = cfg["spatial"]["num_layers"]
    alpha, sigma_2 = inf["alpha"], inf["sigma_2"]
    use_prior = inf.get("use_prior", True)
    info_kw = dict(deconvs=deconvs, alpha=alpha, sigma_2=sigma_2, num_layers=num_layers, use_prior=use_prior)
    settle_kw = dict(
        r_init=r_init,
        deconvs=deconvs,
        alpha=alpha,
        lr_r=inf["lr_r"],
        sigma_2=sigma_2,
        num_epochs_inner=inf["num_epochs_inner"],
        num_layers=num_layers,
        init_noise=inf.get("init_noise", 0.01),
        use_prior=use_prior,
    )

    seq = seq.float()
    if seq.ndim == 3:
        seq = seq.unsqueeze(1)
    T = seq.shape[0]

    encoder = make_settle_encoder(deconvs, r_init, cfg, warm=True)
    encoder.reset()
    r_true, I_true = [], []
    for t in range(T):
        I = _prepare_frame(seq[t])
        I_true.append(I)
        r_true.append(encoder(I))

    hidden = temporal_nn.init_hidden(r_init)
    r_prev = None
    steps = []
    temporal_nn.eval()
    with torch.no_grad():
        for t in range(T):
            r_in = [torch.zeros_like(ri) for ri in r_true[0]] if r_prev is None else r_prev
            r_pred, hidden, _ = temporal_nn(r_in, hidden)
            if t >= 1:
                I_hat = f_clamp(deconvs[0](r_pred[0]))
                I_hat = I_hat - I_hat.mean()
                r_faded, _ = settle_grounded(I_hat, r_warm=r_pred, **settle_kw)
                e_true = settle_info(I_true[t], r_true[t], **info_kw)
                e_faded = settle_info(I_hat, r_faded, **info_kw)
                e_pred = settle_info(I_hat, r_pred, **info_kw)
                mass_t, peak_t = _contrast(I_true[t])
                mass_p, peak_p = _contrast(I_hat)
                steps.append(
                    {
                        "t": t,
                        "closed_loop": t >= split_point,
                        "energy_true": e_true["total_energy"],
                        "energy_faded": e_faded["total_energy"],
                        "energy_pred_code": e_pred["total_energy"],
                        "mass_true": mass_t,
                        "mass_pred": mass_p,
                        "peak_true": peak_t,
                        "peak_pred": peak_p,
                        "cos_pred_true": pair_stats(r_pred, r_true[t])["cos"],
                        "r_pred_norm": float(flat_r(r_pred).norm().item()),
                        "r_true_norm": float(flat_r(r_true[t]).norm().item()),
                    }
                )
            hidden = [hi.detach() for hi in hidden]
            r_prev = clone_r(r_pred) if t >= split_point else clone_r(r_true[t])

    return {"steps": steps, "split_point": split_point}


def aggregate_energy_fade(per_seq, split_point):
    """Mean/std curves across sequences plus pre/post-split summaries."""
    if not per_seq:
        return {"curves": {}, "pre_split": {}, "post_split": {}}
    keys = [
        "energy_true",
        "energy_faded",
        "energy_pred_code",
        "mass_true",
        "mass_pred",
        "peak_true",
        "peak_pred",
        "cos_pred_true",
        "r_pred_norm",
        "r_true_norm",
    ]
    curves = {}
    for k in keys:
        mu, sd = stack_mean_std([[s[k] for s in seq["steps"]] for seq in per_seq])
        curves[f"{k}_mean"] = mu
        curves[f"{k}_std"] = sd

    def _side(pred):
        vals = {k: [] for k in keys}
        for seq in per_seq:
            for s in seq["steps"]:
                if pred(s):
                    for k in keys:
                        vals[k].append(s[k])
        out = {}
        for k, xs in vals.items():
            arr = torch.tensor(xs, dtype=torch.float64) if xs else torch.tensor([float("nan")])
            out[k] = {
                "mean": float(arr.mean().item()) if xs else float("nan"),
                "std": float(arr.std(unbiased=False).item()) if len(xs) > 1 else 0.0,
                "n": len(xs),
            }
        return out

    pre = _side(lambda s: not s["closed_loop"])
    post = _side(lambda s: s["closed_loop"])
    faded = post["energy_faded"]["mean"] - post["energy_true"]["mean"]
    fade = post["mass_true"]["mean"] - post["mass_pred"]["mean"]
    return {
        "curves": curves,
        "pre_split": pre,
        "post_split": post,
        "split_point": split_point,
        "n_sequences": len(per_seq),
        "energy_gap_post": float(faded),
        "fade_gap_post": float(fade),
        "energy_tracks_fade": bool(fade > 0 and faded > 0),
    }
