import torch
import torch.nn.functional as F

from src.strategies.base import TrainingStrategy


class ScheduledSamplingStrategy(TrainingStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["strategy"]
        self.p_start = s.get("p_start", 1.0)
        self.p_end = s.get("p_end", 0.0)
        self.include_tf_loss = s.get("include_tf_loss", True)
        self.n_epochs = cfg["temporal"]["epochs"]

    def teacher_force_prob(self, epoch):
        if self.n_epochs <= 1:
            return self.p_end
        frac = min(max(epoch / (self.n_epochs - 1), 0.0), 1.0)
        return self.p_start + (self.p_end - self.p_start) * frac

    def compute_loss(self, model, ae, latents, frames, epoch):
        t_len = frames.shape[1]
        n_steps = t_len - self.context
        p = self.teacher_force_prob(epoch)

        r_hat_tf = model.predict_latent(latents, teacher_force=True)
        i_hat_tf = ae.decode(r_hat_tf)
        pix_tf = F.mse_loss(i_hat_tf, frames[:, 2:])
        loss_tf = F.mse_loss(r_hat_tf, latents[:, 2:]) + pix_tf

        r_prev, r_curr, hidden = model.init_hidden_from_context(latents[:, : self.context])
        preds = []
        for t in range(n_steps):
            pair = torch.cat([r_prev, r_curr], dim=1).unsqueeze(1)
            r_next, hidden = model._predict_from_pairs(pair, hidden)
            r_next = r_next[:, 0]
            preds.append(r_next)
            use_true = torch.rand((), device=latents.device) < p
            feed = latents[:, self.context + t] if use_true else r_next
            r_prev, r_curr = r_curr, feed

        r_hat_ss = torch.stack(preds, dim=1)
        i_hat_ss = ae.decode(r_hat_ss)
        loss_ss = F.mse_loss(r_hat_ss, latents[:, self.context :]) + F.mse_loss(
            i_hat_ss, frames[:, self.context :]
        )
        loss = loss_tf + loss_ss if self.include_tf_loss else loss_ss
        return {
            "loss": loss,
            "logs": {
                "tf": float(loss_tf.item()),
                "tf_pixel": float(pix_tf.item()),
                "ss": float(loss_ss.item()),
                "p_tf": float(p),
                "n_ar": int(n_steps),
            },
        }
