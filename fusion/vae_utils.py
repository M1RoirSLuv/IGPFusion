from diffusers import AutoencoderKL
import torch
from typing import Dict, Union

from fusion.testload import load_vae_from_ckpt_with_report, try_load_diffusers_vae
from fusion.train_dual_vae_fusion import validate_vae_roundtrip_color


def encode_vae(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    return z * sf


def decode_vae(
    vae: AutoencoderKL,
    z: torch.Tensor,
    decoder_adapter=None,
    adapter_cond: Union[torch.Tensor, Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    vae_dtype = next(vae.parameters()).dtype
    z_scaled = (z / sf).to(dtype=vae_dtype)

    if decoder_adapter is None or adapter_cond is None:
        return vae.decode(z_scaled).sample

    hooks = []
    up_blocks = getattr(vae.decoder, "up_blocks", None)
    if up_blocks is None:
        return vae.decode(z_scaled).sample

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return output
            return decoder_adapter.apply(layer_idx, output, adapter_cond)

        return hook

    for i, block in enumerate(up_blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    try:
        return vae.decode(z_scaled).sample
    finally:
        for h in hooks:
            h.remove()


def load_sd_vae(cfg, logger, device: torch.device) -> AutoencoderKL:
    if cfg.vis_vae_dir:
        vis_vae = try_load_diffusers_vae(cfg.vis_vae_dir)
        if vis_vae is not None:
            logger.info("Using VIS VAE from --vis_vae_dir: %s", cfg.vis_vae_dir)
            return vis_vae.to(device)
    logger.info("Using strict converted ckpt VIS VAE loading from testload.py")
    return load_vae_from_ckpt_with_report(cfg.sd_ckpt_path, threshold=cfg.vae_threshold).to(device)


def maybe_validate_vis_vae(vae_vis: AutoencoderKL, cfg, logger, device: torch.device, output_dir: str):
    if cfg.vis_check_image:
        logger.info("Running VIS VAE roundtrip gate: %s", cfg.vis_check_image)
        validate_vae_roundtrip_color(
            vae_vis,
            vis_image_path=cfg.vis_check_image,
            output_dir=output_dir,
            device=device.type,
        )
