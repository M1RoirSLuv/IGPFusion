from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex"):
        super().__init__()
        self.available = False
        self.lpips = None
        try:
            import lpips

            self.lpips = lpips.LPIPS(net=net)
            self.lpips.requires_grad_(False)
            self.available = True
        except Exception:
            self.available = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # LPIPS expects RGB in [-1,1]
        sr_rgb = sr.repeat(1, 3, 1, 1) * 2 - 1
        hr_rgb = hr.repeat(1, 3, 1, 1) * 2 - 1
        if self.available and self.lpips is not None:
            return self.lpips(sr_rgb, hr_rgb).mean()
        # safe fallback
        return F.l1_loss(sr_rgb, hr_rgb)


def l1_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sr, hr)


def freq_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """DifIISR-inspired frequency amplitude matching loss."""
    sr_fft = torch.fft.rfft2(sr, norm="ortho")
    hr_fft = torch.fft.rfft2(hr, norm="ortho")
    return F.l1_loss(torch.abs(sr_fft), torch.abs(hr_fft))


class DistillLoss(nn.Module):
    def __init__(self, stu_channels: List[int], tea_channels: List[int], proj_ch: int = 64):
        super().__init__()
        n = min(len(stu_channels), len(tea_channels))
        self.n = n
        self.stu_proj = nn.ModuleList([nn.Conv2d(stu_channels[i], proj_ch, 1) for i in range(n)])
        self.tea_proj = nn.ModuleList([nn.Conv2d(tea_channels[i], proj_ch, 1) for i in range(n)])

    def forward(self, stu_feats: List[torch.Tensor], tea_feats: List[torch.Tensor]) -> torch.Tensor:
        n = min(self.n, len(stu_feats), len(tea_feats))
        if n == 0:
            return torch.tensor(0.0, device=stu_feats[0].device if len(stu_feats) else "cpu")

        loss = 0.0
        for i in range(n):
            sf = self.stu_proj[i](stu_feats[i])
            tf = self.tea_proj[i](tea_feats[i])
            if sf.shape[-2:] != tf.shape[-2:]:
                tf = F.interpolate(tf, size=sf.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + F.mse_loss(sf, tf)
        return loss / n
