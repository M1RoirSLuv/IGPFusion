from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADCA(nn.Module):
    """Lightweight channel attention block inspired by AD-CA style design."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(ch // r, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ADSA(nn.Module):
    """Lightweight spatial attention block inspired by AD-SA."""

    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class MCRMBlock(nn.Module):
    """Multi-constraint residual module (lightweight)."""

    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            ADCA(ch),
            ADSA(ch),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class CoRPLELite(nn.Module):
    """Standalone IR SR student network for x2/x4 upscaling."""

    def __init__(self, in_ch: int = 1, feat_ch: int = 64, num_blocks: int = 8, upscale: int = 4):
        super().__init__()
        assert upscale in (2, 4), "Only x2/x4 is supported in this version."
        self.upscale = upscale

        self.head = nn.Conv2d(in_ch, feat_ch, 3, padding=1)
        self.blocks = nn.ModuleList([MCRMBlock(feat_ch) for _ in range(num_blocks)])
        self.fuse = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)

        up_layers = []
        steps = 1 if upscale == 2 else 2
        for _ in range(steps):
            up_layers.extend([
                nn.Conv2d(feat_ch, feat_ch * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ])
        self.upsampler = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(feat_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)

        f = self.head(x)
        feats: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            f = blk(f)
            if i in (1, len(self.blocks) // 2, len(self.blocks) - 1):
                feats.append(f)

        f = self.fuse(f)
        f = self.upsampler(f)
        out = self.tail(f) + base
        out = out.clamp(0.0, 1.0)

        if return_feats:
            return out, feats
        return out
