from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .denoise_net import LatentDenoiser
from .respace import create_gaussian_diffusion


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)


# ===================================================================
# Original one-shot building blocks (preserved for backward compat)
# ===================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * 0.2


class GradientGuidanceBlock(nn.Module):
    def __init__(self, feat_channels: int):
        super().__init__()
        self.img_proj = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.grad_proj = nn.Conv2d(2, feat_channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, 1),
            nn.GELU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        grad = image_gradient(lr)
        grad = F.interpolate(grad, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gi = self.grad_proj(grad)
        ii = self.img_proj(feat)
        return feat + self.fuse(torch.cat([ii, gi], dim=1))


class PromptTokenAdapter(nn.Module):
    """CoRPLE-style learnable prompt tokens, injected on top of CLIP text embeddings."""

    def __init__(self, hidden_dim: int, num_tokens: int = 8):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        b = text_emb.shape[0]
        tok = self.proj(self.tokens).unsqueeze(0).expand(b, -1, -1)
        return torch.cat([tok, text_emb], dim=1)


class PromptEncoder(nn.Module):
    def __init__(self, clip_path: str, use_token_adapter: bool, adapter_tokens: int = 8):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        hidden = self.text_encoder.config.hidden_size
        self.adapter = PromptTokenAdapter(hidden, adapter_tokens) if use_token_adapter else None

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            text = self.text_encoder(**tokens).last_hidden_state
        if self.adapter is not None:
            text = self.adapter(text)
        return text


class DiffusionPriorExtractor(nn.Module):
    def __init__(
        self,
        vae_path: str,
        diffusion_model_path: str,
        clip_path: str,
        use_token_adapter: bool,
        adapter_tokens: int,
        timestep: int = 500,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
        self.unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet", local_files_only=True)
        self.prompt_encoder = PromptEncoder(clip_path, use_token_adapter, adapter_tokens)
        self.timestep = timestep

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()

    def forward(self, lr: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        rgb = lr.repeat(1, 3, 1, 1) * 2.0 - 1.0
        with torch.no_grad():
            z = self.vae.encode(rgb).latent_dist.sample() * self.vae.config.scaling_factor
            t = torch.full((z.shape[0],), self.timestep, device=z.device, dtype=torch.long)
        text_cond = self.prompt_encoder(prompts, z.device).to(z.dtype)
        prior = self.unet(z, t, encoder_hidden_states=text_cond).sample
        return prior


# ===================================================================
# Original one-shot config & model (preserved)
# ===================================================================

@dataclass
class DiffusionPriorConfig:
    in_channels: int = 1
    feat_channels: int = 64
    prior_channels: int = 4
    num_blocks: int = 10
    upscale: int = 4
    vae_path: str = ""
    diffusion_model_path: str = ""
    clip_path: str = "model/clip-vit-large-patch14"
    prior_timestep: int = 500
    use_prompt_adapter: bool = True
    adapter_tokens: int = 8


class DiffusionPriorSR(nn.Module):
    def __init__(self, cfg: DiffusionPriorConfig):
        super().__init__()
        if cfg.upscale not in (2, 4):
            raise ValueError("Only x2/x4 upscale is supported.")
        if not cfg.vae_path:
            raise ValueError("vae_path is required.")
        if not cfg.diffusion_model_path:
            raise ValueError("diffusion_model_path is required.")

        self.upscale = cfg.upscale
        self.default_prompt = "a high quality infrared image with clear thermal edges"
        self.prior = DiffusionPriorExtractor(
            cfg.vae_path,
            cfg.diffusion_model_path,
            cfg.clip_path,
            cfg.use_prompt_adapter,
            cfg.adapter_tokens,
            cfg.prior_timestep,
        )

        self.head = nn.Conv2d(cfg.in_channels, cfg.feat_channels, 3, padding=1)
        self.prior_proj = nn.Conv2d(cfg.prior_channels, cfg.feat_channels, 1)
        self.grad_guide = GradientGuidanceBlock(cfg.feat_channels)

        self.body = nn.Sequential(*[ResidualBlock(cfg.feat_channels) for _ in range(cfg.num_blocks)])
        self.fuse = nn.Conv2d(cfg.feat_channels, cfg.feat_channels, 3, padding=1)

        steps = 1 if cfg.upscale == 2 else 2
        up = []
        for _ in range(steps):
            up += [
                nn.Conv2d(cfg.feat_channels, cfg.feat_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ]
        self.upsample = nn.Sequential(*up)
        self.tail = nn.Conv2d(cfg.feat_channels, 1, 3, padding=1)

    def _normalize_prompts(self, lr: torch.Tensor, prompts: List[str] | None) -> List[str]:
        if prompts is None:
            return [self.default_prompt] * lr.shape[0]
        if len(prompts) == 1 and lr.shape[0] > 1:
            return prompts * lr.shape[0]
        if len(prompts) != lr.shape[0]:
            raise ValueError(f"Prompt count ({len(prompts)}) must match batch size ({lr.shape[0]}).")
        return prompts

    def forward(
        self,
        lr: torch.Tensor,
        prompts: List[str] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        prompts = self._normalize_prompts(lr, prompts)

        prior = self.prior(lr, prompts)
        prior = F.interpolate(prior, size=lr.shape[-2:], mode="bilinear", align_corners=False)

        x = self.head(lr)
        x = x + self.prior_proj(prior)
        x = self.grad_guide(x, lr)
        x = self.body(x)
        x = self.fuse(x)
        x = self.upsample(x)
        sr = self.tail(x).clamp(0.0, 1.0)

        if return_aux:
            return sr, {"prior": prior, "lr_grad": image_gradient(lr), "prompts": prompts}
        return sr


# ===================================================================
# NEW: Iterative Diffusion SR Pipeline (DifIISR-style)
# ===================================================================

@dataclass
class DiffusionSRConfig:
    """Configuration for the iterative diffusion SR pipeline."""
    # Image / latent
    in_channels: int = 1                    # 1 for grayscale IR
    latent_channels: int = 4                # VAE latent channels
    upscale: int = 4

    # VAE
    vae_path: str = ""

    # Denoising UNet
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0
    cond_lq: bool = True                    # concatenate z_y to z_t as input
    use_gradient_guidance: bool = True

    # Diffusion schedule
    diffusion_steps: int = 15
    schedule_name: str = "exponential"
    schedule_power: float = 0.3
    etas_end: float = 0.99
    min_noise_level: float = 0.04
    kappa: float = 2.0
    predict_type: str = "xstart"            # xstart | epsilon | residual
    scale_factor: float = 1.0
    normalize_input: bool = True
    latent_flag: bool = True
    timestep_respacing: int | None = None   # None = use all steps


class DiffusionSRPipeline(nn.Module):
    """
    Full iterative diffusion SR pipeline for infrared images.

    Architecture:
      1. Frozen VAE encodes LQ -> z_y (latent), decodes z_0 -> SR image
      2. LatentDenoiser operates in latent space: (z_t, z_y, t) -> pred_x0
      3. GaussianDiffusion manages the sampling/inversion loop
      4. Gradient guidance from LQ image injected at each denoising step

    Training:
      - Encode HQ to z_x, LQ to z_y
      - Sample random t, add noise to z_x: z_t = q(z_t | z_x, z_y)
      - Denoiser predicts z_x from (z_t, z_y, t)
      - MSE loss in latent space

    Inference (DDIM sampling):
      - Encode LQ -> z_y
      - z_T ~ N(z_y, kappa^2 * eta_T)
      - For t = T-1, ..., 0:
          z_{t-1} = DDIM_step(denoiser, z_t, z_y, t)
      - Decode z_0 -> SR image
    """

    def __init__(self, cfg: DiffusionSRConfig):
        super().__init__()
        if not cfg.vae_path:
            raise ValueError("vae_path is required.")

        self.cfg = cfg
        self.upscale = cfg.upscale

        # --- Frozen VAE ---
        self.vae = AutoencoderKL.from_pretrained(cfg.vae_path, local_files_only=True)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # --- Trainable denoising network ---
        self.denoiser = LatentDenoiser(
            latent_channels=cfg.latent_channels,
            model_channels=cfg.model_channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            dropout=cfg.dropout,
            cond_lq=cfg.cond_lq,
            use_gradient_guidance=cfg.use_gradient_guidance,
        )

        # --- Diffusion process (not nn.Module, just a utility object) ---
        self.diffusion = create_gaussian_diffusion(
            schedule_name=cfg.schedule_name,
            schedule_kwargs={"power": cfg.schedule_power},
            sf=cfg.upscale,
            min_noise_level=cfg.min_noise_level,
            steps=cfg.diffusion_steps,
            kappa=cfg.kappa,
            etas_end=cfg.etas_end,
            weighted_mse=False,
            predict_type=cfg.predict_type,
            timestep_respacing=cfg.timestep_respacing,
            scale_factor=cfg.scale_factor,
            normalize_input=cfg.normalize_input,
            latent_flag=cfg.latent_flag,
        )

    # ------------------------------------------------------------------
    # VAE helpers (grayscale IR -> 3ch RGB -> latent)
    # ------------------------------------------------------------------

    def _to_rgb(self, gray: torch.Tensor) -> torch.Tensor:
        """Convert 1-ch grayscale [0,1] to 3-ch RGB [-1,1]."""
        return gray.repeat(1, 3, 1, 1) * 2.0 - 1.0

    def encode_to_latent(self, img: torch.Tensor, up_sample: bool = False) -> torch.Tensor:
        """Encode image to VAE latent space."""
        if up_sample:
            img = F.interpolate(img, scale_factor=self.upscale, mode="bicubic")
        rgb = self._to_rgb(img)
        with torch.no_grad():
            z = self.vae.encode(rgb.to(self.vae.dtype)).latent_dist.sample()
            z = z * self.vae.config.scaling_factor
        if self.cfg.scale_factor is not None:
            z = z * self.cfg.scale_factor
        return z.to(img.dtype)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VAE latent to image, return grayscale [0,1]."""
        z_dec = z
        if self.cfg.scale_factor is not None:
            z_dec = z_dec / self.cfg.scale_factor
        z_dec = z_dec / self.vae.config.scaling_factor
        with torch.no_grad():
            out = self.vae.decode(z_dec.to(self.vae.dtype)).sample
        # RGB [-1,1] -> grayscale [0,1]
        gray = out.mean(dim=1, keepdim=True).to(z.dtype)
        return (gray * 0.5 + 0.5).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Denoiser wrapper (handles z_y concatenation + lq passing)
    # ------------------------------------------------------------------

    def _build_denoiser_fn(self, z_y: torch.Tensor, lq: torch.Tensor | None = None):
        """
        Return a callable  f(z_t_normalized, t, **kwargs) -> prediction
        that the GaussianDiffusion can call directly.

        Concatenates z_y to the (already-normalized) z_t before passing
        to the denoiser, and forwards the original LQ for gradient guidance.
        """
        denoiser = self.denoiser
        cond_lq = self.cfg.cond_lq

        class _DenoiserWrapper:
            def __call__(self_, x, t, **kwargs):
                if cond_lq:
                    # x is normalized z_t; concatenate z_y
                    x_cat = torch.cat([x, z_y], dim=1)
                else:
                    x_cat = x
                return denoiser(x_cat, t, lq=lq)

        return _DenoiserWrapper()

    # ------------------------------------------------------------------
    # Training forward: compute diffusion loss
    # ------------------------------------------------------------------

    def forward_train(
        self,
        hr: torch.Tensor,
        lr: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute diffusion training loss.

        Args:
            hr: high-quality images [B, 1, H, W], range [0, 1]
            lr: low-quality images [B, 1, h, w], range [0, 1]
            noise: optional pre-generated noise

        Returns:
            dict with 'loss', 'mse', 'z_t', 'pred_zstart'
        """
        B = lr.shape[0]
        device = lr.device

        # Encode to latent space
        z_y = self.encode_to_latent(lr, up_sample=True)   # [B, 4, H/8, W/8]
        z_x = self.encode_to_latent(hr, up_sample=False)  # [B, 4, H/8, W/8]

        # Random timestep
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device).long()

        # Forward diffuse: q(z_t | z_x, z_y)
        if noise is None:
            noise = torch.randn_like(z_x)
        z_t = self.diffusion.q_sample(z_x, z_y, t, noise=noise)

        # Build denoiser fn with conditioning
        model_fn = self._build_denoiser_fn(z_y, lq=lr)

        # Model prediction
        model_output = model_fn(self.diffusion._scale_input(z_t, t), t)

        # Compute target
        from .gaussian_diffusion import ModelMeanType, _extract_into_tensor, mean_flat

        target_map = {
            ModelMeanType.START_X: z_x,
            ModelMeanType.RESIDUAL: z_y - z_x,
            ModelMeanType.EPSILON: noise,
            ModelMeanType.EPSILON_SCALE: noise * self.diffusion.kappa * _extract_into_tensor(
                self.diffusion.sqrt_etas, t, noise.shape
            ),
        }
        target = target_map[self.diffusion.model_mean_type]

        mse = mean_flat((target - model_output) ** 2)

        if self.diffusion.model_mean_type == ModelMeanType.EPSILON_SCALE:
            mse = mse / (self.diffusion.kappa ** 2 * _extract_into_tensor(self.diffusion.etas, t, t.shape))

        loss = mse.mean()

        # Recover pred_zstart for monitoring
        with torch.no_grad():
            if self.diffusion.model_mean_type == ModelMeanType.START_X:
                pred_zstart = model_output.detach()
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON:
                pred_zstart = self.diffusion._predict_xstart_from_eps(
                    x_t=z_t, y=z_y, t=t, eps=model_output.detach()
                )
            elif self.diffusion.model_mean_type == ModelMeanType.RESIDUAL:
                pred_zstart = self.diffusion._predict_xstart_from_residual(
                    y=z_y, residual=model_output.detach()
                )
            elif self.diffusion.model_mean_type == ModelMeanType.EPSILON_SCALE:
                pred_zstart = self.diffusion._predict_xstart_from_eps_scale(
                    x_t=z_t, y=z_y, t=t, eps=model_output.detach()
                )
            else:
                pred_zstart = model_output.detach()

        return {
            "loss": loss,
            "mse": mse.mean().item(),
            "z_t": z_t.detach(),
            "pred_zstart": pred_zstart,
        }

    # ------------------------------------------------------------------
    # DDIM sampling inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(
        self,
        lr: torch.Tensor,
        noise: torch.Tensor | None = None,
        progress: bool = False,
        one_step: bool = False,
    ) -> torch.Tensor:
        """
        Full DDIM sampling: LQ -> SR.

        Args:
            lr: low-quality image [B, 1, h, w], range [0, 1]
            noise: optional initial noise
            progress: show tqdm progress bar
            one_step: if True, single-step prediction (fast, lower quality)

        Returns:
            SR image [B, 1, H, W], range [0, 1]
        """
        z_y = self.encode_to_latent(lr, up_sample=True)
        model_fn = self._build_denoiser_fn(z_y, lq=lr)

        # Initialize z_T ~ N(z_y, kappa^2 * eta_T)
        if noise is None:
            noise = torch.randn_like(z_y)
        z_sample = self.diffusion.prior_sample(z_y, noise)

        # DDIM loop: T-1, ..., 0
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="DDIM sampling")

        for i in indices:
            t = torch.tensor([i] * lr.shape[0], device=lr.device)
            out = self.diffusion.ddim_sample(
                model=model_fn,
                x=z_sample,
                y=z_y,
                t=t,
                clip_denoised=False,
            )
            if one_step:
                z_sample = out["pred_xstart"]
                break
            z_sample = out["sample"]

        # Decode to image
        return self.decode_from_latent(z_sample)

    # ------------------------------------------------------------------
    # DDPM sampling inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_ddpm(
        self,
        lr: torch.Tensor,
        noise: torch.Tensor | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Full DDPM sampling (stochastic, slower than DDIM)."""
        z_y = self.encode_to_latent(lr, up_sample=True)
        model_fn = self._build_denoiser_fn(z_y, lq=lr)

        if noise is None:
            noise = torch.randn_like(z_y)
        z_sample = self.diffusion.prior_sample(z_y, noise)

        indices = list(range(self.diffusion.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="DDPM sampling")

        for i in indices:
            t = torch.tensor([i] * lr.shape[0], device=lr.device)
            out = self.diffusion.p_sample(
                model=model_fn,
                x=z_sample,
                y=z_y,
                t=t,
                clip_denoised=False,
            )
            z_sample = out["sample"]

        return self.decode_from_latent(z_sample)

    # ------------------------------------------------------------------
    # DDIM inversion: encode HQ image to z_T
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_inversion(
        self,
        hr: torch.Tensor,
        lr: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDIM inversion: encode HQ image into the noise space z_T.

        Useful for:
          - Distillation training (teacher model provides z_T targets)
          - Editing / manipulation in latent space

        Args:
            hr: high-quality image [B, 1, H, W], range [0, 1]
            lr: low-quality image [B, 1, h, w], range [0, 1]

        Returns:
            z_T: inverted latent code [B, C_lat, H_lat, W_lat]
        """
        z_y = self.encode_to_latent(lr, up_sample=True)
        z_x = self.encode_to_latent(hr, up_sample=False)
        model_fn = self._build_denoiser_fn(z_y, lq=lr)

        z_sample = z_x
        indices = list(range(1, self.diffusion.num_timesteps))

        for i in indices:
            t = torch.tensor([i] * lr.shape[0], device=lr.device)
            out = self.diffusion.ddim_inverse(
                model=model_fn,
                x=z_sample,
                y=z_y,
                t=t,
            )
            z_sample = out["sample"]

        return z_sample

    # ------------------------------------------------------------------
    # Convenience forward (defaults to training mode)
    # ------------------------------------------------------------------

    def forward(self, hr: torch.Tensor, lr: torch.Tensor, **kwargs) -> dict:
        """Training forward pass. Use sample_ddim() / sample_ddpm() for inference."""
        return self.forward_train(hr, lr, **kwargs)
