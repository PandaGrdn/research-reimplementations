import torch.nn.functional as F

from src.strategies.base import TrainingStrategy


class TeacherForcingStrategy(TrainingStrategy):
    def compute_loss(self, model, ae, latents, frames, epoch):
        r_hat = model.predict_latent(latents, teacher_force=True)
        i_hat = ae.decode(r_hat)
        pix = F.mse_loss(i_hat, frames[:, 2:])
        loss = F.mse_loss(r_hat, latents[:, 2:]) + pix
        return {
            "loss": loss,
            "logs": {
                "tf": float(loss.item()),
                "tf_pixel": float(pix.item()),
                "n_ar": 0,
            },
        }
