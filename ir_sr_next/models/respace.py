"""
Timestep respacing for diffusion sampling.
Allows using fewer timesteps than training while reusing the same model.
Adapted from DifIISR.
"""

import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps: int, sample_timesteps: int) -> set:
    """Create a set of equally-spaced timestep indices from the original schedule."""
    all_steps = [
        int((num_timesteps / sample_timesteps) * x) for x in range(sample_timesteps)
    ]
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that skips steps from a base schedule.

    This lets you train with N steps but sample with fewer.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["sqrt_etas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        new_sqrt_etas = []
        for ii, etas_current in enumerate(base_diffusion.sqrt_etas):
            if ii in self.use_timesteps:
                new_sqrt_etas.append(etas_current)
                self.timestep_map.append(ii)
        kwargs["sqrt_etas"] = np.array(new_sqrt_etas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)


class _WrappedModel:
    """Remaps model timestep indices to the original schedule."""

    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)


def create_gaussian_diffusion(
    *,
    schedule_name: str = "exponential",
    schedule_kwargs: dict | None = None,
    sf: int = 4,
    min_noise_level: float = 0.04,
    steps: int = 15,
    kappa: float = 2.0,
    etas_end: float = 0.99,
    weighted_mse: bool = False,
    predict_type: str = "xstart",
    timestep_respacing: int | None = None,
    scale_factor: float | None = 1.0,
    normalize_input: bool = True,
    latent_flag: bool = True,
):
    """
    Factory function to create a (Spaced)GaussianDiffusion instance.
    Mirrors DifIISR's create_gaussian_diffusion_test.
    """
    from .gaussian_diffusion import get_named_eta_schedule, ModelMeanType, LossType

    sqrt_etas = get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps=steps,
        min_noise_level=min_noise_level,
        etas_end=etas_end,
        kappa=kappa,
        kwargs=schedule_kwargs or {},
    )

    if timestep_respacing is None:
        timestep_respacing = steps

    model_mean_type = {
        "xstart": ModelMeanType.START_X,
        "epsilon": ModelMeanType.EPSILON,
        "epsilon_scale": ModelMeanType.EPSILON_SCALE,
        "residual": ModelMeanType.RESIDUAL,
    }[predict_type]

    loss_type = LossType.WEIGHTED_MSE if weighted_mse else LossType.MSE

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=loss_type,
        scale_factor=scale_factor,
        normalize_input=normalize_input,
        sf=sf,
        latent_flag=latent_flag,
    )
