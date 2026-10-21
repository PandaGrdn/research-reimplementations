"""Extra video datasets, in the same format as Moving MNIST.

Every dataset here yields uint8 clips shaped (N, T, 1, 64, 64), cached as one
file under the data root so a cluster job never regenerates or re-decodes them.

bouncing_balls  Synthetic: balls with elastic wall and ball-ball collisions
                (Sutskever et al., 2009 style). Generated in numpy, no download.
                Different dynamics from Moving MNIST: collisions, no occlusion.
kth             KTH human actions (Schuldt et al., 2004): real grayscale video,
                6 actions x 25 people x 4 scenarios. Frames are center-cropped
                to 120x120 and resized to 64x64, the usual video-prediction
                preprocessing. Split by person (1-16 train/val, 17-25 test) so
                no test person is seen in training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

SIZE = 64

BOUNCING_BALLS_DEFAULTS = {
    "n_sequences": 2000,
    "n_frames": 20,
    "n_balls": 3,
    "radius": 6.0,
    "speed_min": 2.0,
    "speed_max": 4.0,
    "substeps": 4,
    "gen_seed": 1234,
}

KTH_ACTIONS = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
KTH_URL = "https://www.csc.kth.se/cvap/actions/{action}.zip"
KTH_DEFAULTS = {
    "n_frames": 20,
    "frame_stride": 1,
    "clips_per_video": 4,
    "train_max_person": 16,
}


def _key(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]


def _atomic_save(path, save_fn):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    save_fn(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Bouncing balls
# ---------------------------------------------------------------------------

def _initial_state(rng, n, b, r, vmin, vmax):
    pos = rng.uniform(r, SIZE - r, size=(n, b, 2))
    for _ in range(1000):  # resample sequences whose balls start overlapping
        d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
        d[:, np.arange(b), np.arange(b)] = np.inf
        bad = (d < 2 * r + 1).any(axis=(1, 2))
        if not bad.any():
            break
        pos[bad] = rng.uniform(r, SIZE - r, size=(int(bad.sum()), b, 2))
    speed = rng.uniform(vmin, vmax, size=(n, b))
    angle = rng.uniform(0, 2 * np.pi, size=(n, b))
    vel = np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=-1)
    return pos, vel


def _step(pos, vel, r, dt):
    pos = pos + vel * dt
    lo, hi = r, SIZE - r
    below, above = pos < lo, pos > hi
    pos = np.where(below, 2 * lo - pos, pos)
    pos = np.where(above, 2 * hi - pos, pos)
    vel = np.where(below | above, -vel, vel)
    b = pos.shape[1]
    for i in range(b):
        for j in range(i + 1, b):
            d = pos[:, j] - pos[:, i]
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            hit = (dist < 2 * r)[:, 0]
            if not hit.any():
                continue
            n = d / np.maximum(dist, 1e-6)
            u = np.sum((vel[:, i] - vel[:, j]) * n, axis=-1, keepdims=True)
            approaching = (u > 0)[:, 0] & hit
            # equal-mass elastic collision: exchange the velocity along the line of centers
            vel[approaching, i] -= (u * n)[approaching]
            vel[approaching, j] += (u * n)[approaching]
            push = ((2 * r - dist) / 2 * n)[hit]
            pos[hit, i] -= push
            pos[hit, j] += push
    return pos, vel


def _render(pos, r):
    """Anti-aliased discs: intensity 1 inside, linear 1-px edge, max over balls."""
    c = np.arange(SIZE) + 0.5
    yy, xx = np.meshgrid(c, c, indexing="ij")
    dist = np.sqrt((xx[None, None] - pos[:, :, 0, None, None]) ** 2 + (yy[None, None] - pos[:, :, 1, None, None]) ** 2)
    img = np.clip(r + 0.5 - dist, 0.0, 1.0).max(axis=1)
    return np.round(img * 255).astype(np.uint8)


def generate_bouncing_balls(params):
    p = {**BOUNCING_BALLS_DEFAULTS, **(params or {})}
    rng = np.random.default_rng(p["gen_seed"])
    n, t, b, r = p["n_sequences"], p["n_frames"], p["n_balls"], float(p["radius"])
    out = np.zeros((n, t, 1, SIZE, SIZE), dtype=np.uint8)
    chunk = 250
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        pos, vel = _initial_state(rng, m, b, r, p["speed_min"], p["speed_max"])
        for f in range(t):
            out[s:s + m, f, 0] = _render(pos, r)
            for _ in range(p["substeps"]):
                pos, vel = _step(pos, vel, r, 1.0 / p["substeps"])
    return out


def load_bouncing_balls(root, params=None, log=print):
    p = {**BOUNCING_BALLS_DEFAULTS, **(params or {})}
    path = Path(root) / "BouncingBalls" / f"bouncing_balls_{_key(p)}.npy"
    if not path.exists():
        log(f"generating bouncing balls {p} -> {path}")
        clips = generate_bouncing_balls(p)
        _atomic_save(path, lambda tmp: np.save(open(tmp, "wb"), clips))
    return torch.from_numpy(np.load(path)), None


# ---------------------------------------------------------------------------
# KTH
# ---------------------------------------------------------------------------

def _download(url, dest, log=print):
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    os.replace(tmp, dest)


def _decode_avi(path):
    import cv2  # only KTH needs a video decoder

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = g.shape
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        g = cv2.resize(g[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, SIZE, SIZE), np.uint8)


def _video_clips(frames, n_frames, stride, per_video):
    span = (n_frames - 1) * stride + 1
    if len(frames) < span:
        return []
    starts = np.linspace(0, len(frames) - span, per_video).round().astype(int)
    return [frames[s:s + span:stride] for s in sorted(set(starts.tolist()))]


def build_kth(root, params=None, log=print):
    p = {**KTH_DEFAULTS, **(params or {})}
    raw_dir = Path(root) / "KTH" / "raw"
    clips, persons, actions = [], [], []
    for a_idx, action in enumerate(KTH_ACTIONS):
        zpath = raw_dir / f"{action}.zip"
        if not zpath.exists():
            _download(KTH_URL.format(action=action), zpath, log=log)
        with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as td:
            for name in sorted(n for n in zf.namelist() if n.endswith(".avi")):
                person = int(Path(name).name.split("_")[0].replace("person", ""))
                local = zf.extract(name, td)
                frames = _decode_avi(local)
                os.remove(local)
                for clip in _video_clips(frames, p["n_frames"], p["frame_stride"], p["clips_per_video"]):
                    clips.append(clip[:, None])
                    persons.append(person)
                    actions.append(a_idx)
        log(f"kth: {action} done, {len(clips)} clips so far")
    return np.stack(clips).astype(np.uint8), np.array(persons), np.array(actions)


def load_kth(root, params=None, log=print):
    p = {**KTH_DEFAULTS, **(params or {})}
    key_params = {k: p[k] for k in ("n_frames", "frame_stride", "clips_per_video")}
    path = Path(root) / "KTH" / f"kth_clips_{_key(key_params)}.npz"
    if not path.exists():
        clips, persons, actions = build_kth(root, p, log=log)
        _atomic_save(path, lambda tmp: np.savez(open(tmp, "wb"), clips=clips, persons=persons, actions=actions))
    z = np.load(path)
    return torch.from_numpy(z["clips"]), z["persons"]


LOADERS = {"bouncing_balls": load_bouncing_balls, "kth": load_kth}


def dataset_spec(data_cfg):
    """What identifies the data a dictionary was trained on, for cache keys.
    None for Moving MNIST so every existing cache key stays valid."""
    name = (data_cfg or {}).get("name", "moving_mnist")
    if name == "moving_mnist":
        return None
    defaults = {"bouncing_balls": BOUNCING_BALLS_DEFAULTS, "kth": KTH_DEFAULTS}[name]
    return {"name": name, **defaults, **((data_cfg or {}).get(name) or {})}
