from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ir_sr_next.models import DiffusionPriorConfig, DiffusionPriorSR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]["model"]

    model = DiffusionPriorSR(
        DiffusionPriorConfig(
            in_channels=cfg.get("in_channels", 1),
            feat_channels=cfg.get("feat_channels", 64),
            prior_channels=cfg.get("prior_channels", 4),
            num_blocks=cfg.get("num_blocks", 10),
            upscale=cfg.get("upscale", 4),
            vae_path=cfg["vae_path"],
            diffusion_model_path=cfg["diffusion_model_path"],
            clip_path=cfg.get("clip_path", "model/clip-vit-large-patch14"),
            prior_timestep=cfg.get("prior_timestep", 500),
            use_prompt_adapter=bool(cfg.get("use_prompt_adapter", True)),
            adapter_tokens=int(cfg.get("adapter_tokens", 8)),
        )
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    inp = Image.open(args.input).convert("L")
    x = TF.to_tensor(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x).squeeze(0).cpu().clamp(0, 1)

    out = TF.to_pil_image(y)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
