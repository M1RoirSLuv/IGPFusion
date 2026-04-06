from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ir_sr_project.models.corple_student import CoRPLELite


def load_gray_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    x = TF.to_tensor(img).unsqueeze(0)
    return x


def save_gray_tensor(x: torch.Tensor, path: str):
    x = x.squeeze(0).squeeze(0).clamp(0, 1)
    img = TF.to_pil_image(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    cfg = state.get("config", {})
    mcfg = cfg.get("model", {"in_ch": 1, "feat_ch": 64, "num_blocks": 8, "upscale": 4})

    model = CoRPLELite(**mcfg)
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    lr = load_gray_tensor(args.input).to(device)
    with torch.no_grad():
        sr = model(lr)
    save_gray_tensor(sr.cpu(), args.output)
    print(f"Saved SR to {args.output}")


if __name__ == "__main__":
    main()
