from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ir_sr_next.dataset import DatasetConfig, InfraredSRDataset
from ir_sr_next.models import DiffusionPriorConfig, DiffusionPriorSR, image_gradient


class LPIPSLoss(torch.nn.Module):
    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips

        self.loss_fn = lpips.LPIPS(net=net)
        self.loss_fn.requires_grad_(False)
        self.loss_fn.eval()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_rgb = sr.repeat(1, 3, 1, 1) * 2.0 - 1.0
        hr_rgb = hr.repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.loss_fn(sr_rgb, hr_rgb).mean()


def psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = torch.mean((sr - hr) ** 2).item() + 1e-12
    return float(10.0 * np.log10(1.0 / mse))


def freq_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    sr_fft = torch.fft.rfft2(sr, norm="ortho")
    hr_fft = torch.fft.rfft2(hr, norm="ortho")
    return (sr_fft.real - hr_fft.real).abs().mean() + (sr_fft.imag - hr_fft.imag).abs().mean()


def build_loader(cfg: dict, split: str, batch: int, workers: int) -> DataLoader:
    ds = InfraredSRDataset(
        DatasetConfig(
            root=cfg["root"],
            split=cfg[f"{split}_split"],
            hr_subdir=cfg.get("hr_subdir", "HR"),
            lr_subdir=cfg.get("lr_subdir", "LR"),
            exts=tuple(cfg.get("exts", [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])),
            use_precomputed_lr=bool(cfg.get("use_precomputed_lr", True)),
            scale=int(cfg["scale"]),
            hr_size=int(cfg.get("hr_size", 256)),
        )
    )
    return DataLoader(ds, batch_size=batch, shuffle=(split == "train"), num_workers=workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg["runtime"].get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(cfg["data"], "train", cfg["train"]["batch_size"], cfg["data"]["num_workers"])
    val_loader = build_loader(cfg["data"], "val", 1, cfg["data"]["num_workers"])

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(enabled=bool(cfg["runtime"].get("amp", True)))

    use_lpips = float(cfg["loss"].get("w_lpips", 0.0)) > 0
    lpips_loss = LPIPSLoss(net=cfg["loss"].get("lpips_net", "alex")).to(device) if use_lpips else None

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[SR] Epoch {epoch}")
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            pcfg = cfg.get("prompt", {})
            prompts = [pcfg.get("train_prompt", "a high quality infrared image with clear thermal edges")] * lr.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(cfg["runtime"].get("amp", True))):
                sr, aux = model(lr, prompts=prompts, return_aux=True)

                loss_pix = (sr - hr).abs().mean()
                loss_grad = (image_gradient(sr) - image_gradient(hr)).abs().mean()

                prior_hr = model.prior(hr, prompts)
                prior_hr = torch.nn.functional.interpolate(prior_hr, size=aux["prior"].shape[-2:], mode="bilinear", align_corners=False)
                loss_prior = (aux["prior"] - prior_hr).abs().mean()

                loss_freq = freq_loss(sr, hr)
                loss_lp = torch.zeros_like(loss_pix)
                if use_lpips and lpips_loss is not None:
                    with autocast(enabled=False):
                        loss_lp = lpips_loss(sr.float(), hr.float())

                loss = (
                    float(cfg["loss"].get("w_pix", 1.0)) * loss_pix
                    + float(cfg["loss"].get("w_grad", 0.3)) * loss_grad
                    + float(cfg["loss"].get("w_prior", 0.1)) * loss_prior
                    + float(cfg["loss"].get("w_freq", 0.05)) * loss_freq
                    + float(cfg["loss"].get("w_lpips", 0.0)) * loss_lp
                )

            scaler.scale(loss).backward()
            if float(cfg["train"].get("grad_clip", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=float(loss.item()), pix=float(loss_pix.item()), grad=float(loss_grad.item()))

        if epoch % int(cfg["train"].get("val_every", 1)) == 0:
            model.eval()
            scores = []
            with torch.no_grad():
                for batch in val_loader:
                    lr = batch["lr"].to(device)
                    hr = batch["hr"].to(device)
                    pcfg = cfg.get("prompt", {})
                    val_prompt = pcfg.get("val_prompt", pcfg.get("train_prompt", "a high quality infrared image with clear thermal edges"))
                    sr = model(lr, prompts=[val_prompt])
                    scores.append(psnr(sr, hr))

            mean_psnr = float(np.mean(scores)) if scores else 0.0
            ckpt = {
                "epoch": epoch,
                "psnr": mean_psnr,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(ckpt, out_dir / "last.pth")
            if mean_psnr > best:
                best = mean_psnr
                torch.save(ckpt, out_dir / "best.pth")
            print(f"[Val] epoch={epoch} psnr={mean_psnr:.3f}")


if __name__ == "__main__":
    main()
