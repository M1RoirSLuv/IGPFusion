from __future__ import annotations

import torch


class GaussianDiffusion:
    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = arr.to(t.device)[t].float()
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
        ) / self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, model, x_t: torch.Tensor, t: torch.Tensor, model_kwargs: dict | None = None):
        model_kwargs = model_kwargs or {}
        eps = model(x_t, t, **model_kwargs)
        x0 = self.predict_xstart_from_eps(x_t, t, eps)

        coef1 = self._extract(
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            t,
            x_t.shape,
        )
        coef2 = self._extract(
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod),
            t,
            x_t.shape,
        )
        mean = coef1 * x0 + coef2 * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0, eps
