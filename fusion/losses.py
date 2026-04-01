from typing import Tuple

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

from .color import rgb_to_gray


def make_ir_saliency_mask(ir_gray: torch.Tensor, tau: float, temp: float) -> torch.Tensor:
    ir01 = (ir_gray + 1.0) * 0.5
    temp = max(float(temp), 1e-6)
    return torch.sigmoid((ir01 - float(tau)) / temp)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_flat, y_flat = x.flatten(1), y.flatten(1)
    x_center, y_center = x_flat - x_flat.mean(dim=1, keepdim=True), y_flat - y_flat.mean(dim=1, keepdim=True)
    num = (x_center * y_center).sum(dim=1)
    den = torch.sqrt((x_center.square().sum(dim=1) + eps) * (y_center.square().sum(dim=1) + eps))
    return (num / den).mean()


def decomposition_loss(phi_id, phi_vd, phi_ib, phi_vb, epsilon: float):
    l_cdc = pearson_corr(phi_id, phi_vd).abs()
    l_cbc = pearson_corr(phi_ib, phi_vb).clamp(min=0.0)
    l_decomp = (l_cdc**2) / (l_cbc + epsilon)
    return l_decomp, l_cdc, l_cbc


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 3:
        x = rgb_to_gray(x)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


# def make_haze_mask(vis_gray: torch.Tensor, grad_vis: torch.Tensor, brightness_tau: float, grad_tau: float, temp: float) -> torch.Tensor:
#     """Soft haze mask: bright + low-gradient regions are more likely haze."""
#     vis01 = (vis_gray + 1.0) * 0.5
#     temp = max(float(temp), 1e-6)
#     bright = torch.sigmoid((vis01 - float(brightness_tau)) / temp)
#     smooth = torch.sigmoid((float(grad_tau) - grad_vis) / temp)
#     return (bright * smooth).clamp(0.0, 1.0)


# def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#     return (mask * (pred - target).abs()).sum() / (mask.sum() + 1e-6)

def ssim_loss(pred, target):
    # ssim 范围 [0, 1]，1 表示完全一致
    return 1.0 - ssim(pred, target, data_range=2.0) # 因为你的 VAE 输出是 [-1, 1]


def saliency_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss focused on salient regions.

    Args:
        pred: prediction tensor, shape [N, C, H, W].
        target: target tensor with same shape as pred.
        mask: saliency mask, shape [N, 1, H, W], values in [0, 1].

    Returns:
        Scalar masked L1 loss.
    """
    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.repeat(1, pred.size(1), 1, 1)
    return (mask * (pred - target).abs()).sum() / (mask.sum() + 1e-6)
