import hashlib
from pathlib import Path

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset


def prep_seq(seq):
    x = seq.float() / 255.0
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return torch.stack([fr - fr.mean() for fr in x], dim=0)


def _ensure_moving_mnist(root):
    root = Path(root)
    npy = root / "MovingMNIST" / "mnist_test_seq.npy"
    if not npy.exists():
        datasets.MovingMNIST._URL = (
            "https://huggingface.co/datasets/goodgoodideas/STDN/resolve/main/mnist_test_seq.npy"
        )
    return datasets.MovingMNIST(root=str(root), split=None, download=True)


def load_splits(root, n_train, n_val, n_test, seed=0, data_cfg=None):
    name = (data_cfg or {}).get("name", "moving_mnist")
    if name != "moving_mnist":
        return _load_other_splits(name, root, n_train, n_val, n_test, seed, data_cfg)
    raw = _ensure_moving_mnist(root)
    n_total = n_train + n_val + n_test
    if n_total > len(raw):
        raise ValueError(f"need {n_total} sequences, dataset has {len(raw)}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(raw), generator=g)[:n_total].tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]
    train = [prep_seq(raw[i]) for i in train_idx]
    val = [prep_seq(raw[i]) for i in val_idx]
    test = [prep_seq(raw[i]) for i in test_idx]
    info = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "T": int(train[0].shape[0]),
        "shape": tuple(int(x) for x in train[0].shape),
        "split_hash": split_hash(train, val, test),
        "indices_hash": indices_hash(train_idx + val_idx + test_idx),
        "seed": seed,
    }
    return train, val, test, info


def split_indices(n_items, groups, n_train, n_val, n_test, seed, max_train_group=16, name="dataset"):
    """Seeded train/val/test indices. With `groups` (KTH: person id), train and
    val come only from groups <= max_train_group and test only from the rest."""
    g = torch.Generator().manual_seed(seed)
    if groups is None:  # i.i.d. sequences: one random split, as for Moving MNIST
        n_total = n_train + n_val + n_test
        if n_total > n_items:
            raise ValueError(f"{name}: need {n_total} sequences, dataset has {n_items}")
        perm = torch.randperm(n_items, generator=g)[:n_total].tolist()
        return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:n_total]
    groups = [int(x) for x in groups]
    pool_tr = [i for i, p in enumerate(groups) if p <= max_train_group]
    pool_te = [i for i, p in enumerate(groups) if p > max_train_group]
    if n_train + n_val > len(pool_tr) or n_test > len(pool_te):
        raise ValueError(
            f"{name}: need {n_train + n_val} train/val and {n_test} test clips, have {len(pool_tr)} and {len(pool_te)}"
        )
    ptr = torch.randperm(len(pool_tr), generator=g).tolist()
    pte = torch.randperm(len(pool_te), generator=g).tolist()
    return (
        [pool_tr[i] for i in ptr[:n_train]],
        [pool_tr[i] for i in ptr[n_train:n_train + n_val]],
        [pool_te[i] for i in pte[:n_test]],
    )


def _load_other_splits(name, root, n_train, n_val, n_test, seed, data_cfg):
    from src.datasets import LOADERS

    params = (data_cfg or {}).get(name) or {}
    clips, groups = LOADERS[name](root, params)
    train_idx, val_idx, test_idx = split_indices(
        len(clips), groups, n_train, n_val, n_test, seed, params.get("train_max_person", 16), name
    )
    train = [prep_seq(clips[i]) for i in train_idx]
    val = [prep_seq(clips[i]) for i in val_idx]
    test = [prep_seq(clips[i]) for i in test_idx]
    info = {
        "dataset": name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "T": int(train[0].shape[0]),
        "shape": tuple(int(x) for x in train[0].shape),
        "split_hash": split_hash(train, val, test),
        "indices_hash": indices_hash(train_idx + val_idx + test_idx),
        "seed": seed,
    }
    return train, val, test, info


def split_hash(train, val, test):
    h = hashlib.sha256()
    for seqs in (train, val, test):
        h.update(str(len(seqs)).encode())
        if seqs:
            h.update(seqs[0].numpy().tobytes())
            h.update(seqs[-1].numpy().tobytes())
    return h.hexdigest()[:16]


def indices_hash(indices):
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in indices).encode())
    return h.hexdigest()[:16]


def frame_loader(seqs, batch_size, shuffle, seed=0):
    frames = torch.cat(seqs, dim=0)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        TensorDataset(frames),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
    )


def consecutive_pair_loader(seqs, batch_size, shuffle, seed=0):
    prev, nxt = [], []
    for seq in seqs:
        prev.append(seq[:-1])
        nxt.append(seq[1:])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.cat(prev, dim=0), torch.cat(nxt, dim=0)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
    )


def sample_pretrain_frames(seqs, n_frames, seed=0):
    frames = torch.cat(seqs, dim=0)
    n = min(n_frames, frames.shape[0])
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(frames.shape[0], generator=g)[:n]
    return [frames[i] for i in idx]
