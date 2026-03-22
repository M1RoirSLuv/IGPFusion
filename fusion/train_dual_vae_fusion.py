import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint
from PIL import Image
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # skimage 不是硬依赖，缺失时回退到近似实现
    skimage_ssim = None

logger = logging.getLogger(__name__)



def _extract_vae_state_dict(
    state_dict: Dict[str, torch.Tensor],
    vae_state_dict_keys: Optional[set] = None,
    prefix: str = "first_stage_model.",
) -> Dict[str, torch.Tensor]:
    """仅提取 SD checkpoint 中 first_stage_model.* 的 VAE 参数。"""
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        stripped_key = key[len(prefix) :]
        if vae_state_dict_keys is None or stripped_key in vae_state_dict_keys:
            extracted[stripped_key] = value

    logger.info(
        "Extracted VAE keys with prefix '%s': %d", prefix, len(extracted)
    )
    if len(extracted) == 0:
        raise RuntimeError(
            f"No VAE params extracted with prefix '{prefix}'. "
            "Please check checkpoint mapping."
        )
    return extracted



def _first_n(items, n: int = 20):
    return list(items)[:n]



def load_sd_vae_from_ckpt(
    ckpt_path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    prefer_diffusers_vae_dir: Optional[str] = None,
    missing_unexpected_threshold: int = 10,
) -> AutoencoderKL:
    """
    优先从 diffusers VAE 目录加载；若不可用，再从 SD ckpt 提取 first_stage_model 参数。
    并对 missing/unexpected 做严格阈值校验。
    """
    if prefer_diffusers_vae_dir and os.path.isdir(prefer_diffusers_vae_dir):
        logger.info("Loading VIS VAE from diffusers dir: %s", prefer_diffusers_vae_dir)
        vae = AutoencoderKL.from_pretrained(
            prefer_diffusers_vae_dir,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        return vae

    logger.info("Loading VAE weights from checkpoint: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        sample_size=512,
        layers_per_block=2,
    )

    logger.info("AutoencoderKL config: layers_per_block=%s, block_out_channels=%s", vae.config.layers_per_block, vae.config.block_out_channels)

    # 先用 diffusers 官方转换逻辑做 LDM->diffusers VAE 键名映射
    extracted_vae_dict = convert_ldm_vae_checkpoint(state_dict, vae.config)
    logger.info("Converted VAE keys (ldm->diffusers): %d", len(extracted_vae_dict))

    # 兜底：如果转换结果异常为空，再退回纯前缀提取逻辑
    if len(extracted_vae_dict) == 0:
        logger.warning("Converted VAE keys is empty, fallback to raw prefix extraction")
        extracted_vae_dict = _extract_vae_state_dict(state_dict, set(vae.state_dict().keys()))

    incompatible = vae.load_state_dict(extracted_vae_dict, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys

    logger.warning("Missing keys count: %d", len(missing))
    logger.warning("Unexpected keys count: %d", len(unexpected))
    logger.warning("First missing keys (<=20): %s", _first_n(missing, 20))
    logger.warning("First unexpected keys (<=20): %s", _first_n(unexpected, 20))

    if len(missing) > missing_unexpected_threshold or len(unexpected) > missing_unexpected_threshold:
        raise RuntimeError(
            "VAE state_dict strict-check failed: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"threshold={missing_unexpected_threshold}."
        )

    return vae.to(device=device, dtype=dtype)



def _compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))



def _compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if skimage_ssim is not None:
        return float(skimage_ssim(target, pred, data_range=255))

    pred_f = pred.astype(np.float32)
    target_f = target.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x, mu_y = pred_f.mean(), target_f.mean()
    sigma_x, sigma_y = pred_f.var(), target_f.var()
    sigma_xy = ((pred_f - mu_x) * (target_f - mu_y)).mean()
    return float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))



def validate_vae_roundtrip_color(
    vae: AutoencoderKL,
    vis_image_path: str,
    output_dir: str,
    *,
    device: str = "cuda",
    psnr_threshold: float = 18.0,
    ssim_threshold: float = 0.55,
) -> Tuple[float, float]:
    """对 VIS 图做 encode->decode，保存输入/重建并基于 PSNR/SSIM 做阻断校验。"""
    os.makedirs(output_dir, exist_ok=True)
    vae.eval()

    image = Image.open(vis_image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    x = transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(x).latent_dist.mean
        recon = vae.decode(latents).sample.clamp(-1, 1)

    input_np = ((x[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    recon_np = ((recon[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    input_path = os.path.join(output_dir, "vae_roundtrip_input.png")
    recon_path = os.path.join(output_dir, "vae_roundtrip_recon.png")
    Image.fromarray(input_np).save(input_path)
    Image.fromarray(recon_np).save(recon_path)

    input_gray = np.array(Image.fromarray(input_np).convert("L"))
    recon_gray = np.array(Image.fromarray(recon_np).convert("L"))

    psnr = _compute_psnr(recon_gray, input_gray)
    ssim = _compute_ssim(recon_gray, input_gray)

    logger.info("VAE roundtrip saved: input=%s, recon=%s", input_path, recon_path)
    logger.info("VAE roundtrip metrics: PSNR=%.4f, SSIM=%.4f", psnr, ssim)

    if psnr < psnr_threshold or ssim < ssim_threshold:
        raise RuntimeError(
            "VAE roundtrip validation failed: "
            f"PSNR={psnr:.4f} (th={psnr_threshold}), "
            f"SSIM={ssim:.4f} (th={ssim_threshold})."
        )

    return psnr, ssim
