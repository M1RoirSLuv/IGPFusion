import argparse
import logging
import os
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint
from safetensors.torch import load_file as load_safetensors

from fusion.train_dual_vae_fusion import validate_vae_roundtrip_color

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("testload")


def inspect_extract_vae_keys(state_dict: Dict[str, torch.Tensor], prefix: str = "first_stage_model.") -> Dict[str, torch.Tensor]:
    extracted = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            extracted[k[len(prefix):]] = v

    logger.info("总key数量: %d", len(state_dict))
    logger.info("以 '%s' 开头的VAE key数量: %d", prefix, len(extracted))
    if extracted:
        sample_keys = list(extracted.keys())[:20]
        logger.info("前20个提取key示例: %s", sample_keys)
    else:
        logger.error("未提取到任何VAE key，请检查checkpoint格式")
    return extracted


def load_vae_from_ckpt_with_report(
    ckpt_path: str,
    threshold: int = 10,
) -> AutoencoderKL:
    logger.info("开始读取checkpoint: %s", ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_path, device="cpu")
    else:
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

    logger.info("当前VAE配置: layers_per_block=%s, block_out_channels=%s", vae.config.layers_per_block, vae.config.block_out_channels)

    extracted = inspect_extract_vae_keys(state_dict)
    # baseline: 不做转换，直接按key交集匹配（用于对照）
    filtered_direct = {k: v for k, v in extracted.items() if k in vae.state_dict()}
    logger.info("[对照] 直接匹配到AutoencoderKL的key数量: %d", len(filtered_direct))

    # 使用 diffusers 官方转换函数处理 LDM -> diffusers 键名
    converted = convert_ldm_vae_checkpoint(state_dict, vae.config)
    logger.info("[转换] convert_ldm_vae_checkpoint 产出key数量: %d", len(converted))
    if converted:
        logger.info("[转换] 前20个key示例: %s", list(converted.keys())[:20])
    else:
        logger.warning("[转换] 结果为空，将回退到直接匹配结果")

    load_candidate = converted if converted else filtered_direct

    incompatible = vae.load_state_dict(load_candidate, strict=False)
    missing = incompatible.missing_keys
    unexpected = incompatible.unexpected_keys

    logger.info("missing数量: %d", len(missing))
    logger.info("unexpected数量: %d", len(unexpected))
    logger.info("前20个missing: %s", missing[:20])
    logger.info("前20个unexpected: %s", unexpected[:20])

    if len(missing) > threshold or len(unexpected) > threshold:
        raise RuntimeError(
            f"VAE加载异常: missing={len(missing)}, unexpected={len(unexpected)}, threshold={threshold}"
        )

    logger.info("VAE读取通过严格阈值检查")
    return vae


def try_load_diffusers_vae(path: str) -> Optional[AutoencoderKL]:
    if not path:
        return None
    if not os.path.isdir(path):
        logger.warning("diffusers目录不存在，跳过: %s", path)
        return None

    logger.info("尝试从diffusers目录加载VAE: %s", path)
    vae = AutoencoderKL.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32)
    logger.info("diffusers目录VAE加载成功")
    return vae


def main():
    parser = argparse.ArgumentParser(description="仅测试VAE读取与roundtrip，不进行训练")
    parser.add_argument("--ckpt", type=str, default="./model/v1-5-pruned.ckpt")
    parser.add_argument("--diffusers_vae_dir", type=str, default=None)
    parser.add_argument("--vis_image", type=str, default="/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Vis/00000.png", help="用于roundtrip测试的可见光图片路径")
    parser.add_argument("--report_dir", type=str, default="./vae_load_report")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    vae = None
    if args.diffusers_vae_dir:
        try:
            vae = try_load_diffusers_vae(args.diffusers_vae_dir)
        except Exception as e:
            logger.error("diffusers目录加载失败: %s", e)

    if vae is None:
        vae = load_vae_from_ckpt_with_report(args.ckpt, threshold=args.threshold)

    if args.vis_image:
        logger.info("开始roundtrip质量检查: %s", args.vis_image)
        psnr, ssim = validate_vae_roundtrip_color(
            vae.to(args.device),
            vis_image_path=args.vis_image,
            output_dir=args.report_dir,
            device=args.device,
        )
        logger.info("roundtrip结果: PSNR=%.4f, SSIM=%.4f", psnr, ssim)
    else:
        logger.info("未提供--vis_image，跳过roundtrip质量检查")


if __name__ == "__main__":
    main()
