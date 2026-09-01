from abc import ABC, abstractmethod


class TrainingStrategy(ABC):
    """Encapsulates HOW the temporal model is trained each epoch.
    Everything a 'solution' can vary lives here."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.context = cfg["temporal"]["context"]

    @abstractmethod
    def compute_loss(self, model, ae, latents, frames, epoch) -> dict:
        """Returns {'loss': tensor, 'logs': {...}}.
        latents: precomputed encoder outputs [B, T, C, H, W] (frozen AE).
        Must handle burn-in internally (hidden state warmup on context)."""

    def on_epoch_end(self, epoch):
        return None
