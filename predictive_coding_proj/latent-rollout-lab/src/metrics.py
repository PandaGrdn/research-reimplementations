import torch
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    _ssim = None

try:
    import lpips as _lpips_mod
except ImportError:
    _lpips_mod = None


def mse(pred, target):
    return F.mse_loss(pred, target)


def per_step_mse(pred, target):
    err = (pred - target).flatten(2).pow(2).mean(dim=2)
    return err.mean(dim=0)


def latent_l2(pred, target):
    return (pred - target).flatten(2).norm(dim=2).mean(dim=0)


def latent_cos(pred, target):
    a = pred.flatten(2)
    b = target.flatten(2)
    return F.cosine_similarity(a, b, dim=2).mean(dim=0)


def _to_np_frame(x):
    return x.detach().cpu().numpy()


def per_step_ssim(pred, target):
    if _ssim is None:
        return pred.new_zeros(pred.shape[1]).cpu()
    b, t = pred.shape[:2]
    scores = []
    for ti in range(t):
        acc = 0.0
        for bi in range(b):
            p = _to_np_frame(pred[bi, ti, 0])
            g = _to_np_frame(target[bi, ti, 0])
            data_range = float(max(g.max() - g.min(), 1e-6))
            acc += _ssim(g, p, data_range=data_range)
        scores.append(acc / max(b, 1))
    return torch.tensor(scores, dtype=torch.float32)


class LPIPSMeter:
    def __init__(self, device, enabled=True):
        self.enabled = enabled and _lpips_mod is not None
        self.fn = None
        if self.enabled:
            self.fn = _lpips_mod.LPIPS(net="alex").to(device)
            self.fn.eval()
            for p in self.fn.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def per_step(self, pred, target):
        if not self.enabled:
            return pred.new_zeros(pred.shape[1]).cpu()
        b, t = pred.shape[:2]
        scores = []
        for ti in range(t):
            p = pred[:, ti].repeat(1, 3, 1, 1).clamp(-1, 1)
            g = target[:, ti].repeat(1, 3, 1, 1).clamp(-1, 1)
            scores.append(self.fn(p, g).mean().item())
        return torch.tensor(scores, dtype=torch.float32)


def copy_last_frames(frames, context, n_steps):
    last = frames[:, context - 1 : context]
    return last.expand(-1, n_steps, -1, -1, -1).contiguous()


def mean_context_frames(frames, context, n_steps):
    mean = frames[:, :context].mean(dim=1, keepdim=True)
    return mean.expand(-1, n_steps, -1, -1, -1).contiguous()


def headline_mean(curve, start=5, end=18):
    curve = list(curve)
    lo = min(start, len(curve))
    hi = min(end, len(curve))
    if hi <= lo:
        return float("nan") if not curve else float(sum(curve) / len(curve))
    sl = curve[lo:hi]
    return float(sum(sl) / max(len(sl), 1))
