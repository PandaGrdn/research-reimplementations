import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64), kernel_size=4, stride=2, padding=1):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.channels = channels
        self.latent_channels = channels[-1]

        enc = []
        c_in = in_channels
        for c_out in channels:
            enc.append(nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding))
            enc.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.encoder = nn.Sequential(*enc)

        dec = []
        dec_ch = list(reversed(channels))
        for i, c_in in enumerate(dec_ch):
            c_out = dec_ch[i + 1] if i + 1 < len(dec_ch) else in_channels
            dec.append(nn.ConvTranspose2d(c_in, c_out, kernel_size, stride=stride, padding=padding))
            if i + 1 < len(dec_ch):
                dec.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*dec)

    def encode(self, frames):
        lead = frames.shape[:-3]
        x = frames.reshape(-1, *frames.shape[-3:])
        r = self.encoder(x)
        return r.view(*lead, *r.shape[1:])

    def decode(self, r):
        lead = r.shape[:-3]
        x = r.reshape(-1, *r.shape[-3:])
        frames = self.decoder(x)
        return frames.view(*lead, self.in_channels, *frames.shape[-2:])

    def forward(self, frames):
        return self.decode(self.encode(frames))


def ae_channels(cfg):
    a = cfg["ae"]
    if a.get("channels"):
        return list(a["channels"])
    return [a.get("latent_channels", 32)]


def build_ae(cfg):
    a = cfg["ae"]
    return Autoencoder(
        in_channels=a.get("in_channels", 1),
        channels=ae_channels(cfg),
        kernel_size=a.get("kernel_size", 4),
        stride=a.get("stride", 2),
        padding=a.get("padding", 1),
    )
