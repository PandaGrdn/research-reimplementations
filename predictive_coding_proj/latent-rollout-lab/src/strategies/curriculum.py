import torch.nn.functional as F

from src.strategies.base import TrainingStrategy


def unroll_curriculum(epoch, max_unroll=8, warmup=8, epochs_per_stage=3):
    if epoch < warmup:
        return 0
    n = 1
    for _ in range((epoch - warmup) // epochs_per_stage):
        n = min(n * 2, max_unroll)
    return n


class CurriculumStrategy(TrainingStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        s = cfg["strategy"]
        self.warmup = s.get("warmup", 8)
        self.epochs_per_stage = s.get("epochs_per_stage", 3)
        self.max_unroll = s.get("max_unroll")

    def _n_ar(self, epoch, t_len):
        max_unroll = self.max_unroll
        if max_unroll is None:
            max_unroll = t_len - self.context
        return unroll_curriculum(
            epoch,
            max_unroll=max_unroll,
            warmup=self.warmup,
            epochs_per_stage=self.epochs_per_stage,
        )

    def compute_loss(self, model, ae, latents, frames, epoch):
        t_len = frames.shape[1]
        n_ar = min(self._n_ar(epoch, t_len), t_len - self.context)

        r_hat_tf = model.predict_latent(latents, teacher_force=True)
        i_hat_tf = ae.decode(r_hat_tf)
        pix_tf = F.mse_loss(i_hat_tf, frames[:, 2:])
        loss_tf = F.mse_loss(r_hat_tf, latents[:, 2:]) + pix_tf

        if n_ar > 0:
            r_hat_ar = model.rollout(latents[:, : self.context], n_steps=n_ar)
            i_hat_ar = ae.decode(r_hat_ar)
            loss_ar = F.mse_loss(r_hat_ar, latents[:, self.context : self.context + n_ar]) + F.mse_loss(
                i_hat_ar, frames[:, self.context : self.context + n_ar]
            )
            loss = loss_tf + loss_ar
        else:
            loss_ar = frames.new_zeros(())
            loss = loss_tf

        return {
            "loss": loss,
            "logs": {
                "tf": float(loss_tf.item()),
                "tf_pixel": float(pix_tf.item()),
                "ar": float(loss_ar.item()),
                "n_ar": int(n_ar),
            },
        }
