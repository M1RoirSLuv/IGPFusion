from __future__ import annotations

import math

import torch


def make_beta_schedule(schedule: str, num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_steps)
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float32)
        t = steps / num_steps
        alphas_bar = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-6, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")
