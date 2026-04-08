from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from ir_sr_next.diffusion import DDIMInversion, DiffusionIRSampler, GaussianDiffusion, GuidanceConfig, make_beta_schedule, space_timesteps


def encode_text(prompts: list[str], clip_path: str, device: torch.device) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
    text_model = CLIPTextModel.from_pretrained(clip_path, local_files_only=True).to(device)
    text_model.eval()
    tokens = tokenizer(prompts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        text = text_model(**tokens).last_hidden_state
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a high quality infrared image with clear thermal edges")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = cfg["model"]
    dcfg = cfg.get("diffusion", {})

    device = torch.device(cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL.from_pretrained(mcfg["vae_path"], local_files_only=True).to(device)
    unet = UNet2DConditionModel.from_pretrained(mcfg["diffusion_model_path"], subfolder="unet", local_files_only=True).to(device)
    vae.eval()
    unet.eval()

    img = Image.open(args.input).convert("L")
    lr = TF.to_tensor(img).unsqueeze(0).to(device)

    lr_rgb = lr.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        z0 = vae.encode(lr_rgb).latent_dist.sample() * vae.config.scaling_factor

    betas = make_beta_schedule(dcfg.get("beta_schedule", "linear"), int(dcfg.get("num_steps", 1000)))
    diffusion = GaussianDiffusion(betas)
    timesteps = space_timesteps(int(dcfg.get("num_steps", 1000)), dcfg.get("respace", "ddim50"))

    text_cond = encode_text([args.prompt], mcfg.get("clip_path", "model/clip-vit-large-patch14"), device)

    inversion = DDIMInversion(diffusion)
    model_fn = lambda xt, tt, **kw: unet(xt, tt, encoder_hidden_states=kw["text_cond"]).sample
    zt = inversion.invert(model_fn, z0, timesteps, model_kwargs={"text_cond": text_cond})

    sampler = DiffusionIRSampler(
        diffusion,
        vae,
        unet,
        GuidanceConfig(
            w_pix=float(dcfg.get("w_pix", 0.2)),
            w_grad=float(dcfg.get("w_grad", 0.2)),
            step_size=float(dcfg.get("guidance_step", 0.1)),
        ),
    )

    z_sr = sampler.sample(zt, text_cond, timesteps, lr)
    with torch.no_grad():
        sr = vae.decode(z_sr / vae.config.scaling_factor).sample[:, :1]
    sr = ((sr + 1.0) * 0.5).clamp(0, 1)

    out = TF.to_pil_image(sr.squeeze(0).cpu())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"Saved SR: {args.output}")


if __name__ == "__main__":
    main()
