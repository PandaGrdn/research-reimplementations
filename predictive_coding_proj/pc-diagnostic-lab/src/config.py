import copy
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_dicts(base, overlay):
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(paths):
    cfg = {}
    for p in paths:
        cfg = merge_dicts(cfg, load_yaml(p))
    return cfg


def lab_root(start=None):
    return Path(start) if start is not None else Path(__file__).resolve().parents[1]


def resolve_config(overlays=None, smoke=False, root=None):
    root = lab_root(root)
    paths = [
        root / "configs" / "base.yaml",
        root / "configs" / "spatial_pc.yaml",
        root / "configs" / "temporal_pc.yaml",
    ]
    if smoke:
        paths.append(root / "configs" / "smoke.yaml")
    for extra in overlays or []:
        paths.append(Path(extra))
    return load_config(paths)


def scheduled_value(start, end, epoch, n_epochs):
    if end is None:
        return start
    frac = epoch / max(n_epochs - 1, 1)
    return start + (end - start) * frac
