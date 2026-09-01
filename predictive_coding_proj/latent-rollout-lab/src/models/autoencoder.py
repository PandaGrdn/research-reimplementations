import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=32, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.encoder = nn.Conv2d(in_channels, latent_channels, kernel_size, stride=stride, padding=padding)
        self.decoder = nn.ConvTranspose2d(
            latent_channels, in_channels, kernel_size, stride=stride, padding=padding
        )

    def encode(self, frames):
        lead = frames.shape[:-3]
        x = frames.reshape(-1, *frames.shape[-3:])
        r = F.relu(self.encoder(x))
        return r.view(*lead, *r.shape[1:])

    def decode(self, r):
        lead = r.shape[:-3]
        x = r.reshape(-1, *r.shape[-3:])
        frames = self.decoder(x)
        return frames.view(*lead, 1, *frames.shape[-2:])

    def forward(self, frames):
        return self.decode(self.encode(frames))


def build_ae(cfg):
    a = cfg["ae"]
    return Autoencoder(
        in_channels=a.get("in_channels", 1),
        latent_channels=a.get("latent_channels", 32),
        kernel_size=a.get("kernel_size", 4),
        stride=a.get("stride", 2),
        padding=a.get("padding", 1),
    )
