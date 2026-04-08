from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from ir_sr_next.diffusion import DDIMInversion, DiffusionIRSampler, GaussianDiffusion, GuidanceConfig, make_beta_schedule, space_timesteps
from ir_sr_next.models import DiffusionPriorConfig, DiffusionPriorSR


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    model = DiffusionPriorSR(
        DiffusionPriorConfig(
            in_channels=mcfg.get("in_channels", 1),
            feat_channels=mcfg.get("feat_channels", 64),
            prior_channels=mcfg.get("prior_channels", 4),
            num_blocks=mcfg.get("num_blocks", 10),
            upscale=mcfg.get("upscale", 4),
            vae_path=mcfg["vae_path"],
            diffusion_model_path=mcfg["diffusion_model_path"],
            clip_path=mcfg.get("clip_path", "model/clip-vit-large-patch14"),
            prior_timestep=mcfg.get("prior_timestep", 500),
            use_prompt_adapter=bool(mcfg.get("use_prompt_adapter", True)),
            adapter_tokens=int(mcfg.get("adapter_tokens", 8)),
        )
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def infer_sr(model: DiffusionPriorSR, img: torch.Tensor, prompt: str) -> torch.Tensor:
    with torch.no_grad():
        y = model(img, prompts=[prompt]).squeeze(0).cpu().clamp(0, 1)
    return y


def infer_ddim(model: DiffusionPriorSR, img: torch.Tensor, cfg: dict, prompt: str) -> torch.Tensor:
    dcfg = cfg.get("diffusion", {})
    betas = make_beta_schedule(dcfg.get("beta_schedule", "linear"), int(dcfg.get("num_steps", 1000)))
    diffusion = GaussianDiffusion(betas)
    timesteps = space_timesteps(int(dcfg.get("num_steps", 1000)), dcfg.get("respace", "ddim50"))

    lr_rgb = img.repeat(1, 3, 1, 1) * 2.0 - 1.0
    with torch.no_grad():
        z0 = model.prior.vae.encode(lr_rgb).latent_dist.sample() * model.prior.vae.config.scaling_factor
        text_cond = model.prior.prompt_encoder([prompt], img.device).to(z0.dtype)

    model_fn = lambda xt, tt, **kw: model.prior.unet(xt, tt, encoder_hidden_states=kw["text_cond"]).sample
    inversion = DDIMInversion(diffusion)
    zt = inversion.invert(model_fn, z0, timesteps, model_kwargs={"text_cond": text_cond})

    sampler = DiffusionIRSampler(
        diffusion,
        model.prior.vae,
        model.prior.unet,
        GuidanceConfig(
            w_pix=float(dcfg.get("w_pix", 0.2)),
            w_grad=float(dcfg.get("w_grad", 0.2)),
            step_size=float(dcfg.get("guidance_step", 0.1)),
        ),
    )

    z_sr = sampler.sample(zt, text_cond, timesteps, img)
    with torch.no_grad():
        sr = model.prior.vae.decode(z_sr / model.prior.vae.config.scaling_factor).sample[:, :1]
    return ((sr + 1.0) * 0.5).squeeze(0).cpu().clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="sr", choices=["sr", "ddim"])
    parser.add_argument("--prompt", type=str, default="a high quality infrared image with clear thermal edges")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)

    inp = Image.open(args.input).convert("L")
    x = TF.to_tensor(inp).unsqueeze(0).to(device)

    if args.mode == "ddim":
        y = infer_ddim(model, x, cfg, args.prompt)
    else:
        y = infer_sr(model, x, args.prompt)

    out = TF.to_pil_image(y)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
