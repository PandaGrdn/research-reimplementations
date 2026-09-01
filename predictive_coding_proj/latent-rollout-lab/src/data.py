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


def load_splits(root, n_train, n_val, n_test, seed=0):
    raw = _ensure_moving_mnist(root)
    n_total = n_train + n_val + n_test
    if n_total > len(raw):
        raise ValueError(f"need {n_total} sequences, dataset has {len(raw)}")
    train = [prep_seq(raw[i]) for i in range(n_train)]
    val = [prep_seq(raw[n_train + i]) for i in range(n_val)]
    test = [prep_seq(raw[n_train + n_val + i]) for i in range(n_test)]
    info = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "T": int(train[0].shape[0]),
        "shape": tuple(int(x) for x in train[0].shape),
        "split_hash": split_hash(train, val, test),
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


def sequence_loader(frames, latents, batch_size, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        TensorDataset(frames, latents),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
    )


@torch.no_grad()
def cache_latents(ae, seqs, device, batch_size=32):
    ae.eval()
    frames = torch.stack(seqs)
    chunks = []
    for i in range(0, frames.shape[0], batch_size):
        batch = frames[i : i + batch_size].to(device)
        chunks.append(ae.encode(batch).cpu())
    return torch.cat(chunks, dim=0)
