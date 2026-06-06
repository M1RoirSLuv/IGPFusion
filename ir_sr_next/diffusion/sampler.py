from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gaussian_diffusion import GaussianDiffusion


@dataclass
class GuidanceConfig:
    w_pix: float = 0.2
    w_grad: float = 0.2
    step_size: float = 0.1


def image_grad(x: torch.Tensor) -> torch.Tensor:
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)


class DiffusionIRSampler:
    """
    Diffusion sampling with per-step guidance gradient injection.
    This is the key mechanism requested for DifIISR-style optimization-in-the-loop.
    """

    def __init__(self, diffusion: GaussianDiffusion, vae, unet, guidance_cfg: GuidanceConfig):
        self.diffusion = diffusion
        self.vae = vae
        self.unet = unet
        self.guidance_cfg = guidance_cfg

    def _decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img = self.vae.decode(z / self.vae.config.scaling_factor).sample
        img = (img[:, :1] + 1.0) * 0.5
        return img.clamp(0.0, 1.0)

    def _guidance_loss(self, x0_pred: torch.Tensor, lr_up: torch.Tensor) -> torch.Tensor:
        pix = (x0_pred - lr_up).abs().mean()
        grad = (image_grad(x0_pred) - image_grad(lr_up)).abs().mean()
        return self.guidance_cfg.w_pix * pix + self.guidance_cfg.w_grad * grad

    def sample(self, z_t: torch.Tensor, text_cond: torch.Tensor, timesteps: list[int], lr_img: torch.Tensor) -> torch.Tensor:
        x = z_t
        lr_up = F.interpolate(lr_img, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)

        for t_idx in reversed(timesteps):
            x = x.detach().requires_grad_(True)
            t = torch.full((x.shape[0],), int(t_idx), device=x.device, dtype=torch.long)
            mean, var, x0, _ = self.diffusion.p_mean_variance(
                lambda xt, tt, **kw: self.unet(xt, tt, encoder_hidden_states=kw["text_cond"]).sample,
                x,
                t,
                model_kwargs={"text_cond": text_cond},
            )

            x0_img = self._decode_latent(x0)
            guide_loss = self._guidance_loss(x0_img, lr_up)
            g = torch.autograd.grad(guide_loss, x, retain_graph=False, create_graph=False, allow_unused=True)[0]
            if g is None:
                g = torch.zeros_like(x)

            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(var.clamp(min=1e-12)) * noise - self.guidance_cfg.step_size * g

        return x
