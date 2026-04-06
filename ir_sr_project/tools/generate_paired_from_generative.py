#!/usr/bin/env python3
"""Generate paired HR/LR infrared data from SD1.5 (+ optional LoRA).

Workflow:
1) Use generative model to synthesize HR image.
2) Apply deterministic degradations to create LR image.
3) Save paired files with aligned names for supervised SR training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paired HR/LR IR data from SD1.5")
    p.add_argument("--base_model", type=str, required=True, help="SD1.5 model path or HF id")
    p.add_argument("--lora_path", type=str, default="", help="Optional LoRA weights path")
    p.add_argument("--prompt_file", type=str, required=True, help="One prompt per line")
    p.add_argument("--out_root", type=str, required=True, help="Output root containing HR/ LR")
    p.add_argument("--num_images_per_prompt", type=int, default=4)
    p.add_argument("--hr_size", type=int, default=512, help="Generated HR size (square)")
    p.add_argument("--scale", type=int, default=4, help="Downsample scale to get LR")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", type=str, default="blurry, noisy, low contrast, artifacts")
    p.add_argument("--blur_sigma", type=float, default=1.2, help="Gaussian blur radius before downsampling")
    p.add_argument("--noise_std", type=float, default=2.0, help="Gaussian noise std in [0,255] after downsampling")
    p.add_argument("--to_grayscale", action="store_true", help="Convert output HR/LR to grayscale")
    return p.parse_args()


def load_prompts(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def degrade_to_lr(hr: Image.Image, scale: int, blur_sigma: float, noise_std: float) -> Image.Image:
    lr_w = max(1, hr.width // scale)
    lr_h = max(1, hr.height // scale)

    x = hr.filter(ImageFilter.GaussianBlur(radius=blur_sigma)) if blur_sigma > 0 else hr
    x = x.resize((lr_w, lr_h), resample=Image.Resampling.BICUBIC)

    if noise_std > 0:
        arr = np.array(x).astype(np.float32)
        arr += np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        x = Image.fromarray(arr)
    return x


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompt_file))
    if not prompts:
        raise ValueError("No valid prompts found.")

    try:
        from diffusers import StableDiffusionPipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Please install diffusers + transformers + accelerate.") from exc

    out_root = Path(args.out_root)
    hr_dir = out_root / "HR"
    lr_dir = out_root / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    pipe = pipe.to(device)

    g = torch.Generator(device=device).manual_seed(args.seed)
    np.random.seed(args.seed)

    idx = 0
    for prompt in prompts:
        for _ in range(args.num_images_per_prompt):
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.hr_size,
                width=args.hr_size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=g,
            )
            hr = result.images[0]
            lr = degrade_to_lr(hr, scale=args.scale, blur_sigma=args.blur_sigma, noise_std=args.noise_std)

            if args.to_grayscale:
                hr = hr.convert("L")
                lr = lr.convert("L")

            name = f"img_{idx:06d}.png"
            hr.save(hr_dir / name)
            lr.save(lr_dir / name)
            idx += 1

    print(f"Saved paired data to {out_root}: HR={idx}, LR={idx}")


if __name__ == "__main__":
    main()
