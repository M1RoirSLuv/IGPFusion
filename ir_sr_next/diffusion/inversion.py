from __future__ import annotations

import torch

from .gaussian_diffusion import GaussianDiffusion


class DDIMInversion:
    """A lightweight deterministic inversion helper."""

    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

    def invert(self, model, x0: torch.Tensor, timesteps: list[int], model_kwargs: dict | None = None) -> torch.Tensor:
        """Map clean latent x0 to noisy latent at final timestep by forward noising along timetable."""
        x = x0
        model_kwargs = model_kwargs or {}
        for t_idx in timesteps:
            t = torch.full((x.shape[0],), int(t_idx), device=x.device, dtype=torch.long)
            eps = model(x, t, **model_kwargs)
            alpha = self.diffusion._extract(self.diffusion.sqrt_alphas_cumprod, t, x.shape)
            sigma = self.diffusion._extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x = alpha * x0 + sigma * eps
        return x
