"""
Gaussian diffusion utilities for IR super-resolution.
Adapted from DifIISR: supports eta-based noise schedule, DDIM sampling/inversion,
and per-timestep LQ-conditioned denoising in latent space.
"""

import enum
import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def get_named_eta_schedule(
    schedule_name: str,
    num_diffusion_timesteps: int,
    min_noise_level: float,
    etas_end: float = 0.99,
    kappa: float = 1.0,
    kwargs: dict | None = None,
) -> np.ndarray:
    """Return sqrt_etas array for the given schedule name."""
    if kwargs is None:
        kwargs = {}
    if schedule_name == "exponential":
        power = kwargs.get("power", 0.3)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(
            1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start)
        )
        base = np.ones([num_diffusion_timesteps]) * increaser
        power_timestep = (
            np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        )
        power_timestep *= num_diffusion_timesteps - 1
        sqrt_etas = np.power(base, power_timestep) * etas_start
    else:
        raise ValueError(f"Unknown schedule_name {schedule_name}")
    return sqrt_etas


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelMeanType(enum.Enum):
    START_X = enum.auto()       # predict x_0
    EPSILON = enum.auto()       # predict epsilon
    RESIDUAL = enum.auto()      # predict y - x_0
    EPSILON_SCALE = enum.auto() # predict scaled epsilon


class LossType(enum.Enum):
    MSE = enum.auto()
    WEIGHTED_MSE = enum.auto()


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array for a batch of indices."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Gaussian diffusion process with eta-based noise schedule.

    Supports:
      - Forward diffusion  q(x_t | x_0, y)
      - DDPM sampling       p(x_{t-1} | x_t)
      - DDIM sampling       (deterministic)
      - DDIM inversion      encode x_0 -> z_T
      - Training losses
      - VAE encode / decode via first_stage_model
    """

    def __init__(
        self,
        *,
        sqrt_etas: np.ndarray,
        kappa: float,
        model_mean_type: ModelMeanType,
        loss_type: LossType,
        sf: int = 4,
        scale_factor: float | None = None,
        normalize_input: bool = True,
        latent_flag: bool = True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        self.sqrt_etas = sqrt_etas.astype(np.float64)
        self.etas = self.sqrt_etas ** 2
        assert len(self.etas.shape) == 1
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa ** 2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
            self.posterior_variance[1], self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas

        # DDIM coefficients
        self.etas_prev_clipped = np.append(self.etas_prev[1], self.etas_prev[1:])
        self.ddim_coef1 = self.etas_prev * self.etas
        self.ddim_coef2 = self.etas_prev / self.etas

        # MSE loss weights
        if model_mean_type in (ModelMeanType.START_X, ModelMeanType.RESIDUAL):
            weight = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas) ** 2
        elif model_mean_type in (ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE):
            weight = (
                0.5
                / self.posterior_variance_clipped
                * (kappa * self.alpha / ((1 - self.etas) * self.sqrt_etas)) ** 2
            )
        else:
            raise NotImplementedError(model_mean_type)
        self.weight_loss_mse = weight

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(self, x_start, y, t, noise=None):
        """Sample from q(x_t | x_0, y)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start)
            + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # Prior sample (initialize z_T)
    # ------------------------------------------------------------------

    def prior_sample(self, y, noise=None):
        """Sample z_T ~ q(x_T | y) = N(y, kappa^2 * eta_T)."""
        if noise is None:
            noise = torch.randn_like(y)
        t = torch.tensor([self.num_timesteps - 1] * y.shape[0], device=y.device).long()
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    # ------------------------------------------------------------------
    # Model output -> predictions
    # ------------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        return (
            x_t
            - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
            - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        return (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        return y - residual

    def _scale_input(self, inputs, t):
        """Normalize noisy input for numerical stability."""
        if self.normalize_input:
            if self.latent_flag:
                std = torch.sqrt(
                    _extract_into_tensor(self.etas, t, inputs.shape) * self.kappa ** 2 + 1
                )
            else:
                inputs_max = (
                    _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                )
                std = inputs_max
            return inputs / std
        return inputs

    # ------------------------------------------------------------------
    # p_mean_variance: core denoising step
    # ------------------------------------------------------------------

    def p_mean_variance(
        self, model, x_t, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t) and predict x_0.

        Returns dict with: mean, variance, log_variance, pred_xstart,
                           ddim_k, ddim_m, ddim_j  (for DDIM update).
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        assert t.shape == (B,)

        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        # DDIM coefficients
        ddim_coef1 = _extract_into_tensor(self.ddim_coef1, t, x_t.shape)
        ddim_coef2 = _extract_into_tensor(self.ddim_coef2, t, x_t.shape)
        etas = _extract_into_tensor(self.etas, t, x_t.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x_t.shape)
        k = 1 - etas_prev + torch.sqrt(ddim_coef1) - torch.sqrt(ddim_coef2)
        m = torch.sqrt(ddim_coef2)
        j = etas_prev - torch.sqrt(ddim_coef1)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )
        else:
            raise ValueError(f"Unknown mean type: {self.model_mean_type}")

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "ddim_k": k,
            "ddim_m": m,
            "ddim_j": j,
        }

    # ------------------------------------------------------------------
    # DDPM sampling
    # ------------------------------------------------------------------

    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Single DDPM sampling step: x_{t-1} from x_t."""
        out = self.p_mean_variance(
            model, x, y, t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self, y, model, first_stage_model=None, noise=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
        device=None, progress=False, one_step=False, apply_decoder=True,
    ):
        """Full DDPM sampling loop: z_T -> z_0 -> decoded image."""
        final = None
        for sample in self.p_sample_loop_progressive(
            y, model,
            first_stage_model=first_stage_model,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            one_step=one_step,
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def p_sample_loop_progressive(
        self, y, model, first_stage_model=None, noise=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
        device=None, progress=False, one_step=False,
    ):
        """DDPM sampling loop, yields intermediate results."""
        if device is None:
            device = next(model.parameters()).device

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        if noise is None:
            noise = torch.randn_like(z_y)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="DDPM sampling")

        for i in indices:
            t = torch.tensor([i] * y.shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model, z_sample, z_y, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    def ddim_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Single DDIM step: deterministic x_{t-1} from x_t."""
        out = self.p_mean_variance(
            model=model, x_t=x, y=y, t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = pred_xstart * out["ddim_k"] + out["ddim_m"] * x + out["ddim_j"] * y
        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_sample_loop(
        self, y, model, first_stage_model=None, noise=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
        device=None, progress=False, one_step=False, apply_decoder=True, zT=None,
    ):
        """Full DDIM sampling loop."""
        final = None
        for sample in self.ddim_sample_loop_progressive(
            y=y, model=model,
            first_stage_model=first_stage_model,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            one_step=one_step,
            zT=zT,
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def ddim_sample_loop_progressive(
        self, y, model, first_stage_model=None, noise=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None,
        device=None, progress=False, one_step=False, zT=None,
    ):
        """DDIM sampling loop, yields intermediate results."""
        if device is None:
            device = next(model.parameters()).device

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        if zT is None:
            z_sample = self.prior_sample(z_y, noise)
        else:
            z_sample = zT

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="DDIM sampling")

        for i in indices:
            t = torch.tensor([i] * z_y.shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model=model, x=z_sample, y=z_y, t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]

    # ------------------------------------------------------------------
    # DDIM inversion: encode HQ image -> z_T
    # ------------------------------------------------------------------

    def ddim_inverse(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """Single DDIM inverse step: x_t from x_{t-1}."""
        out = self.p_mean_variance(
            model, x, y, t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = (x - pred_xstart * out["ddim_k"] - out["ddim_j"] * y) / out["ddim_m"]
        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_inverse_loop(
        self, x, y, model, first_stage_model=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None,
    ):
        """Full DDIM inversion: encode HQ x conditioned on LQ y -> latent z_T."""
        final = None
        for sample in self.ddim_inverse_loop_progressive(
            x, y, model,
            first_stage_model=first_stage_model,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
        ):
            final = sample["sample"]
        return final

    def ddim_inverse_loop_progressive(
        self, x, y, model, first_stage_model=None,
        clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None,
    ):
        """DDIM inversion loop, yields intermediate results."""
        if device is None:
            device = next(model.parameters()).device

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_x = self.encode_first_stage(x, first_stage_model, up_sample=False)

        indices = list(range(1, self.num_timesteps))
        z_sample = z_x

        for i in indices:
            t = torch.tensor([i] * y.shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_inverse(
                    model, z_sample, z_y, t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                z_sample = out["sample"]

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def training_losses(self, model, x_start, y, t, first_stage_model=None, model_kwargs=None, noise=None):
        """
        Compute training loss for a single timestep.

        :param model: denoising model
        :param x_start: HQ images [N, C, H, W]
        :param y: LQ images [N, C, H, W]
        :param t: timestep indices [N]
        :param first_stage_model: VAE for latent encoding
        :param model_kwargs: extra kwargs for model (e.g. lq conditioning)
        :param noise: optional pre-generated noise
        :return: (terms_dict, z_t, pred_zstart)
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = torch.randn_like(z_start)

        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        model_output = model(self._scale_input(z_t, t), t, **model_kwargs)

        terms = {}
        target = {
            ModelMeanType.START_X: z_start,
            ModelMeanType.RESIDUAL: z_y - z_start,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.EPSILON_SCALE: noise * self.kappa * _extract_into_tensor(self.sqrt_etas, t, noise.shape),
        }[self.model_mean_type]

        assert model_output.shape == target.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)

        if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            terms["mse"] /= self.kappa ** 2 * _extract_into_tensor(self.etas, t, t.shape)

        if self.loss_type == LossType.WEIGHTED_MSE:
            weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
        else:
            weights = 1
        terms["loss"] = terms["mse"] * weights

        # Recover pred_zstart for monitoring
        if self.model_mean_type == ModelMeanType.START_X:
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output.detach())
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        """Encode image to latent space via VAE."""
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode="bicubic")
        if first_stage_model is None:
            return y
        with torch.no_grad():
            y = y.to(dtype=next(first_stage_model.parameters()).dtype)
            z_y = first_stage_model.encode(y)
            if hasattr(z_y, "latent_dist"):
                # diffusers AutoencoderKL
                z_y = z_y.latent_dist.sample()
                if hasattr(first_stage_model, "config") and hasattr(first_stage_model.config, "scaling_factor"):
                    z_y = z_y * first_stage_model.config.scaling_factor
            if self.scale_factor is not None:
                z_y = z_y * self.scale_factor
            return z_y.to(dtype=ori_dtype)

    def decode_first_stage(self, z_sample, first_stage_model=None):
        """Decode latent code to image via VAE."""
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        with torch.no_grad():
            if self.scale_factor is not None:
                z_sample = z_sample / self.scale_factor
            if hasattr(first_stage_model, "config") and hasattr(first_stage_model.config, "scaling_factor"):
                z_sample = z_sample / first_stage_model.config.scaling_factor
            z_sample = z_sample.to(dtype=next(first_stage_model.parameters()).dtype)
            out = first_stage_model.decode(z_sample)
            if hasattr(out, "sample"):
                out = out.sample
            return out.to(dtype=ori_dtype)
