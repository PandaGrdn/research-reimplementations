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


def resolve_config(strategy_yaml, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    base = root / "configs" / "base.yaml"
    model = root / "configs" / "model" / "convgru.yaml"
    paths = [base, model, Path(strategy_yaml)]
    return load_config(paths)
