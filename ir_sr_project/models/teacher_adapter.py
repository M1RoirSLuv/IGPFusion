from __future__ import annotations

from contextlib import contextmanager
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherAdapter(nn.Module):
    """
    Lightweight teacher adapter.

    Fallback mode: dependency-free scale-space features.
    Diffusers mode: load UNet (and optional VAE) from a local SD-style folder
    and extract teacher priors from the pretrained infrared generative model.
    """

    def __init__(self, teacher_path: str, layer_keys: List[str]):
        super().__init__()
        self.teacher_path = teacher_path
        self.layer_keys = layer_keys
        self.unet: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.loaded = False
        self._try_load_modules()

    def _try_load_modules(self) -> None:
        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel

            self.unet = UNet2DConditionModel.from_pretrained(self.teacher_path, subfolder="unet")
            self.unet.requires_grad_(False)
            self.unet.eval()

            try:
                self.vae = AutoencoderKL.from_pretrained(self.teacher_path, subfolder="vae")
                self.vae.requires_grad_(False)
                self.vae.eval()
            except Exception:
                self.vae = None

            self.loaded = True
        except Exception:
            self.unet = None
            self.vae = None
            self.loaded = False

    @contextmanager
    def _no_grad_eval(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            yield
        self.train(was_training)

    def _encode_lr_to_latent(self, lr: torch.Tensor) -> torch.Tensor:
        """Encode LR image to latent with VAE when available."""
        if self.vae is None:
            # Fallback latent approximation: resize + channel project.
            z = F.interpolate(lr, scale_factor=1.0, mode="bilinear", align_corners=False)
            return z.repeat(1, 4, 1, 1)

        x = lr
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        h, w = x.shape[-2:]
        h8, w8 = max(8, h - h % 8), max(8, w - w % 8)
        if h8 != h or w8 != w:
            x = F.interpolate(x, size=(h8, w8), mode="bilinear", align_corners=False)

        x = x * 2.0 - 1.0
        latent_dist = self.vae.encode(x).latent_dist
        latents = latent_dist.sample() * getattr(self.vae.config, "scaling_factor", 0.18215)
        return latents

    def extract(self, lr: torch.Tensor) -> List[torch.Tensor]:
        """
        Return multi-level teacher features aligned to student's expected feature count.

        If SD modules are available, teacher features come from the model's latent denoiser.
        Otherwise use simple scale-space features as safe fallback.
        """
        if self.loaded and self.unet is not None:
            with self._no_grad_eval():
                latents = self._encode_lr_to_latent(lr)
                b = latents.shape[0]
                timestep = torch.full((b,), 500, device=latents.device, dtype=torch.long)
                encoder_hidden_states = torch.zeros(
                    (b, 77, self.unet.config.cross_attention_dim),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                feat = self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states).sample
                feat_mid = F.avg_pool2d(feat, 2, 2)
                feat_up = F.interpolate(feat_mid, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            return [feat, feat_mid, feat_up]

        # Dependency-free fallback features.
        x1 = lr
        x2 = F.avg_pool2d(lr, 2, 2)
        x3 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        return [x1, x2, x3]
