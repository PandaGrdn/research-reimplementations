import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def get_device(name="auto"):
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_hash(cwd):
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def pip_freeze():
    try:
        return subprocess.check_output(["pip", "freeze"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return ""


def clone_r(r):
    return [ri.detach().clone() for ri in r]


def zeros_like_r(r):
    return [torch.zeros_like(ri) for ri in r]


def to_device_r(r, device):
    return [ri.to(device) for ri in r]


def mean_std(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def write_yaml(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def new_run_dir(root, exp_name, arm=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{arm}" if arm and arm != "baseline" else ""
    run_dir = Path(root) / "results" / exp_name / f"{stamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = Path(root) / "results" / exp_name / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.is_dir() and not any(latest.iterdir()):
            latest.rmdir()
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        latest.mkdir(parents=True, exist_ok=True)
        (latest / ".target").write_text(str(run_dir))
    return run_dir


def finish_run(run_dir, cfg, metrics, root=None, summary=None):
    run_dir = Path(run_dir)
    write_yaml(run_dir / "config.yaml", cfg)
    env = {
        "git": git_hash(root or run_dir),
        "pip": pip_freeze(),
        "device": str(cfg.get("device")),
        "seed": cfg.get("seed"),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    latest = run_dir.parent / "latest"
    latest_metrics = latest / "metrics.json" if latest.is_dir() and not latest.is_symlink() else None
    if latest_metrics is not None:
        latest_metrics.write_text((run_dir / "metrics.json").read_text())
        write_yaml(latest / "config.yaml", cfg)
    line = summary or metrics.get("summary") or json.dumps({k: metrics[k] for k in metrics if k != "seeds"})
    print(line)
    print(f"wrote {run_dir}")
    return run_dir


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not json serializable: {type(obj)}")


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)
