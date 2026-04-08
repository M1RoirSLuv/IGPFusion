from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)


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
