#!/usr/bin/env python3
"""Generate synthetic infrared HR images with SD1.5 (+ optional LoRA).

This script is optional. For SR training, real HR infrared data is still preferred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_prompts(prompt_file: Path) -> list[str]:
    lines = [ln.strip() for ln in prompt_file.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic IR HR images")
    parser.add_argument("--base_model", type=str, required=True, help="SD1.5 model path or HF id")
    parser.add_argument("--lora_path", type=str, default="", help="Optional LoRA weights path")
    parser.add_argument("--prompt_file", type=str, required=True, help="Text file with one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="blurry, noisy, low contrast, artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(args.prompt_file))
    if not prompts:
        raise ValueError("No valid prompts found in prompt_file.")

    try:
        from diffusers import StableDiffusionPipeline
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Please install diffusers + transformers + accelerate first.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    idx = 0
    for prompt in prompts:
        for _ in range(args.num_images_per_prompt):
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            image.save(output_dir / f"synthetic_hr_{idx:06d}.png")
            idx += 1

    print(f"Saved {idx} images to {output_dir}")


if __name__ == "__main__":
    main()
