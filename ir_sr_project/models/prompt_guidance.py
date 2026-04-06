from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PromptConfig:
    model_name_or_path: str = "model/clip-vit-large-patch14"
    positive_prompt: str = "a high quality infrared image with clear thermal edges"
    negative_prompt: str = "a blurry noisy low quality infrared image"
    margin: float = 0.1


class CLIPPromptLoss(nn.Module):
    """Prompt ranking loss based on frozen CLIP image/text encoders."""

    def __init__(self, cfg: PromptConfig):
        super().__init__()
        self.cfg = cfg
        self.available = False
        self.tokenizer = None
        self.clip = None

        try:
            from transformers import CLIPModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name_or_path, local_files_only=True)
            self.clip = CLIPModel.from_pretrained(cfg.model_name_or_path, local_files_only=True)
            self.clip.requires_grad_(False)
            self.clip.eval()
            self.available = True
        except Exception:
            self.available = False

    def _encode_text(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.tokenizer is not None and self.clip is not None
        texts = [self.cfg.positive_prompt, self.cfg.negative_prompt]
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        txt_feat = self.clip.get_text_features(**tokens)
        txt_feat = F.normalize(txt_feat, dim=-1)
        return txt_feat[0:1], txt_feat[1:2]

    def _preprocess(self, sr: torch.Tensor) -> torch.Tensor:
        # sr: [B,1,H,W] in [0,1]
        sr_rgb = sr.repeat(1, 3, 1, 1)
        sr_rgb = F.interpolate(sr_rgb, size=(224, 224), mode="bicubic", align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=sr.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=sr.device).view(1, 3, 1, 1)
        return (sr_rgb - mean) / std

    def forward(self, sr: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if not self.available or self.clip is None or self.tokenizer is None:
            zero = sr.new_tensor(0.0)
            return zero, {"s_pos": 0.0, "s_neg": 0.0}

        device = sr.device
        with torch.no_grad():
            t_pos, t_neg = self._encode_text(device)

        pixel_values = self._preprocess(sr)
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        img_feat = F.normalize(img_feat, dim=-1)

        s_pos = (img_feat * t_pos).sum(dim=-1)
        s_neg = (img_feat * t_neg).sum(dim=-1)

        loss = torch.relu(self.cfg.margin - s_pos + s_neg).mean()
        stats = {"s_pos": float(s_pos.mean().detach().item()), "s_neg": float(s_neg.mean().detach().item())}
        return loss, stats
