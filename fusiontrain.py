import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from diffusers import AutoencoderKL


LOGGER = logging.getLogger("dual_vae_fusion")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        ],
    )


@dataclass
class TrainConfig:
    # Paths
    sd_ckpt_path: str
    ir_vae_path: str
    ir_data_dir: str
    vis_data_dir: str
    output_dir: str

    # Data / loader
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 4

    # Optim
    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4

    # Schedule
    epochs_stage1: int = 10
    epochs_stage2: int = 30

    # Model
    latent_channels: int = 4
    feature_channels: int = 64

    # Stage-1 loss weights: reconstruction + decomposition
    recon_weight_ir: float = 1.0
    recon_weight_vis: float = 1.0
    decomp_weight_stage1: float = 0.5

    # Stage-2 loss weights: fusion + decomposition
    intensity_weight: float = 1.0
    grad_weight: float = 0.2
    decomp_weight_stage2: float = 0.2
    epsilon_decomp: float = 1.01
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    save_every: int = 1


class PairedInfraredVisibleDataset(Dataset):
    """Dataset for paired infrared/visible images matched by file stem."""

    def __init__(self, ir_dir: str, vis_dir: str, image_size: int):
        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        ir_files = {
            p.stem: p
            for p in self.ir_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        }
        vis_files = {
            p.stem: p
            for p in self.vis_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_ext
        }

        common_stems = sorted(set(ir_files) & set(vis_files))
        if not common_stems:
            raise RuntimeError(
                f"No paired files found. IR dir: {ir_dir}, VIS dir: {vis_dir}. "
                "Pairing expects same filename stem."
            )

        self.pairs: List[Tuple[Path, Path, str]] = [
            (ir_files[s], vis_files[s], s) for s in common_stems
        ]

        self.transform_vis = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # IR image is loaded as grayscale, then repeated to 3 channels for VAE compatibility.
        self.transform_ir = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        LOGGER.info("Loaded %d paired samples.", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ir_path, vis_path, stem = self.pairs[idx]
        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("RGB")

        ir = self.transform_ir(ir_img)
        ir = ir.repeat(3, 1, 1)
        vis = self.transform_vis(vis_img)

        return {
            "ir": ir,
            "vis": vis,
            "stem": stem,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LatentAlign(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 1),
            nn.GroupNorm(8, feat_ch),
            nn.SiLU(),
            ConvBlock(feat_ch, feat_ch),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FeatureDecomposer(nn.Module):
    """Split features into base (low-frequency/shared) and detail (high-frequency/specific)."""

    def __init__(self, channels: int):
        super().__init__()
        self.base_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.detail_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # base: low-frequency shared content; detail: high-frequency residual
        base = self.base_branch(x)
        detail = x - base
        detail = self.detail_refine(detail)
        return base, detail


class LatentReconstructor(nn.Module):
    def __init__(self, feat_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(feat_ch * 2, feat_ch),
            nn.Conv2d(feat_ch, out_ch, 1),
        )

    def forward(self, base: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([base, detail], dim=1))


class FusionNetwork(nn.Module):
    """Fuse two modalities with gated base/detail strategy."""

    def __init__(self, feat_ch: int, latent_ch: int):
        super().__init__()
        self.base_gate = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 1),
            nn.Sigmoid(),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            ConvBlock(feat_ch * 2, feat_ch),
            nn.Conv2d(feat_ch, latent_ch, 1),
        )

    def forward(
        self,
        ir_base: torch.Tensor,
        ir_detail: torch.Tensor,
        vis_base: torch.Tensor,
        vis_detail: torch.Tensor,
    ) -> torch.Tensor:
        base_cat = torch.cat([ir_base, vis_base], dim=1)
        detail_cat = torch.cat([ir_detail, vis_detail], dim=1)

        wb = self.base_gate(base_cat)
        wd = self.detail_gate(detail_cat)

        # Gate controls modality contribution per spatial location/channel.
        base_fused = wb * ir_base + (1.0 - wb) * vis_base
        detail_fused = wd * ir_detail + (1.0 - wd) * vis_detail

        return self.out_proj(torch.cat([base_fused, detail_fused], dim=1))


class DualVAEFusionModel(nn.Module):
    def __init__(self, latent_ch: int, feat_ch: int):
        super().__init__()
        self.align_ir = LatentAlign(latent_ch, feat_ch)
        self.align_vis = LatentAlign(latent_ch, feat_ch)

        self.decomp_ir = FeatureDecomposer(feat_ch)
        self.decomp_vis = FeatureDecomposer(feat_ch)

        self.rec_ir = LatentReconstructor(feat_ch, latent_ch)
        self.rec_vis = LatentReconstructor(feat_ch, latent_ch)

        self.fusion = FusionNetwork(feat_ch, latent_ch)

    def forward_stage1(
        self, z_ir: torch.Tensor, z_vis: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h_ir = self.align_ir(z_ir)
        h_vis = self.align_vis(z_vis)

        phi_ib, phi_id = self.decomp_ir(h_ir)
        phi_vb, phi_vd = self.decomp_vis(h_vis)

        z_ir_rec = self.rec_ir(phi_ib, phi_id)
        z_vis_rec = self.rec_vis(phi_vb, phi_vd)

        return {
            "phi_ib": phi_ib,
            "phi_id": phi_id,
            "phi_vb": phi_vb,
            "phi_vd": phi_vd,
            "z_ir_rec": z_ir_rec,
            "z_vis_rec": z_vis_rec,
        }

    def forward_stage2(
        self, z_ir: torch.Tensor, z_vis: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        stage1_outputs = self.forward_stage1(z_ir, z_vis)
        z_fused = self.fusion(
            stage1_outputs["phi_ib"],
            stage1_outputs["phi_id"],
            stage1_outputs["phi_vb"],
            stage1_outputs["phi_vd"],
        )
        stage1_outputs["z_fused"] = z_fused
        return stage1_outputs


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Per-sample Pearson coefficient over all dims, then mean across batch.
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)

    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)

    num = (x_center * y_center).sum(dim=1)
    den = torch.sqrt((x_center.square().sum(dim=1) + eps) * (y_center.square().sum(dim=1) + eps))
    corr = num / den
    return corr.mean()


def decomposition_loss(
    phi_id: torch.Tensor,
    phi_vd: torch.Tensor,
    phi_ib: torch.Tensor,
    phi_vb: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # LCDC should be small (detail decorrelation), LCBC should be large (base correlation).
    l_cdc = pearson_corr(phi_id, phi_vd).abs()
    l_cbc = pearson_corr(phi_ib, phi_vb).clamp(min=0.0)
    l_decomp = (l_cdc ** 2) / (l_cbc + epsilon)
    return l_decomp, l_cdc, l_cbc


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 3:
        x = rgb_to_gray(x)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


@torch.no_grad()
def encode_vae(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    # VAE encoders are frozen in this training script.
    posterior = vae.encode(x).latent_dist
    z = posterior.sample()
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    return z * sf


def decode_vae(vae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    # Do not use no_grad here: gradients from image-space losses must flow back to z/fusion net.
    # VAE params are still frozen by requires_grad_(False), so only fusion modules are updated.
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    vae_dtype = next(vae.parameters()).dtype
    # Avoid mixed half/float mismatch in conv bias under autocast.
    z = (z / sf).to(dtype=vae_dtype)
    return vae.decode(z).sample


def _torch_load_trusted_checkpoint(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint with Torch>=2.6 compatibility (weights_only default changed)."""
    # Add common safe globals used by older .ckpt formats (e.g. Lightning checkpoints).
    try:
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        torch.serialization.add_safe_globals([type(np.dtype(np.float32))])
        torch.serialization.add_safe_globals([type(np.float64(0.0))])
        torch.serialization.add_safe_globals([type(np.int64(0))])
    except Exception:
        pass

    try:
        import pytorch_lightning.callbacks.model_checkpoint as pl_mc  # type: ignore

        torch.serialization.add_safe_globals([pl_mc.ModelCheckpoint])
    except Exception:
        # Lightning may be unavailable; this is only best-effort allowlisting.
        pass

    # Trusted local checkpoint: use weights_only=False to avoid PyTorch 2.6 default issues.
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unexpected checkpoint type: {type(obj)}")


def _extract_vae_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract VAE weights from common SD ckpt naming conventions."""
    candidates = [
        "first_stage_model.",
        "vae.",
        "model.first_stage_model.",
    ]

    # Prefer prefix-based extraction.
    for prefix in candidates:
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if sub:
            return sub

    # Fallback: if keys already look like AutoencoderKL weights, return as-is.
    vae_hint_keys = [
        "encoder.conv_in.weight",
        "decoder.conv_in.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
    ]
    if any(k in state_dict for k in vae_hint_keys):
        return state_dict

    raise RuntimeError("Could not find VAE weights in checkpoint. Expected prefixes: first_stage_model./vae./model.first_stage_model.")


def load_sd_vae_from_ckpt(ckpt_path: str, image_size: int, device: torch.device) -> AutoencoderKL:
    """Load SD VAE robustly across diffusers/torch checkpoint format differences."""
    try:
        LOGGER.info("Trying AutoencoderKL.from_single_file for %s", ckpt_path)
        vae = AutoencoderKL.from_single_file(
            ckpt_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        return vae.to(device)
    except Exception as e:
        LOGGER.warning("from_single_file failed: %s", e)
        LOGGER.info("Falling back to manual torch.load(weights_only=False) + state_dict extraction.")

    raw_state = _torch_load_trusted_checkpoint(ckpt_path)
    vae_state = _extract_vae_state_dict(raw_state)

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        sample_size=image_size,
    )

    missing, unexpected = vae.load_state_dict(vae_state, strict=False)
    LOGGER.info("Manual VAE load complete. missing=%d unexpected=%d", len(missing), len(unexpected))
    return vae.to(device)


def save_visualization(
    save_dir: Path,
    step_tag: str,
    ir: torch.Tensor,
    vis: torch.Tensor,
    fused: torch.Tensor,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    # Convert from [-1, 1] to [0, 1].
    panel = torch.cat([ir[:1], vis[:1], fused[:1]], dim=0)
    panel = (panel.clamp(-1, 1) + 1.0) / 2.0
    save_image(panel, save_dir / f"{step_tag}.png", nrow=3)


def train(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    setup_logging(output_dir / "logs")
    LOGGER.info("Training configuration: %s", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)

    # Load VAEs. In this script both VAEs are used as fixed feature extractors/decoders.
    LOGGER.info("Loading SD1.5 RGB VAE from ckpt: %s", cfg.sd_ckpt_path)
    vae_vis = load_sd_vae_from_ckpt(cfg.sd_ckpt_path, cfg.image_size, device)

    LOGGER.info("Loading IR VAE from directory: %s", cfg.ir_vae_path)
    vae_ir = AutoencoderKL.from_pretrained(
        cfg.ir_vae_path,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)

    vae_ir.eval().requires_grad_(False)
    vae_vis.eval().requires_grad_(False)

    dataset = PairedInfraredVisibleDataset(cfg.ir_data_dir, cfg.vis_data_dir, cfg.image_size)

    # Keep workers within a conservative bound to avoid dataloader freezes on small nodes.
    cpu_count = os.cpu_count() or 1
    max_recommended_workers = max(0, cpu_count - 1)
    effective_num_workers = min(cfg.num_workers, max_recommended_workers)
    if effective_num_workers != cfg.num_workers:
        LOGGER.warning(
            "num_workers=%d is reduced to %d (cpu_count=%d)",
            cfg.num_workers,
            effective_num_workers,
            cpu_count,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = DualVAEFusionModel(cfg.latent_channels, cfg.feature_channels).to(device)

    stage1_params = (
        list(model.align_ir.parameters())
        + list(model.align_vis.parameters())
        + list(model.decomp_ir.parameters())
        + list(model.decomp_vis.parameters())
        + list(model.rec_ir.parameters())
        + list(model.rec_vis.parameters())
    )

    stage2_params = list(model.parameters())

    opt_stage1 = torch.optim.AdamW(
        stage1_params,
        lr=cfg.lr_stage1,
        weight_decay=cfg.weight_decay,
    )
    opt_stage2 = torch.optim.AdamW(
        stage2_params,
        lr=cfg.lr_stage2,
        weight_decay=cfg.weight_decay,
    )

    global_step = 0

    def run_epoch(epoch: int, stage: int) -> Dict[str, float]:
        nonlocal global_step
        model.train()

        optimizer = opt_stage1 if stage == 1 else opt_stage2

        stats = {
            "loss": 0.0,
            "l_rec_ir": 0.0,
            "l_rec_vis": 0.0,
            "l_int": 0.0,
            "l_grad": 0.0,
            "l_decomp": 0.0,
            "l_cdc": 0.0,
            "l_cbc": 0.0,
        }

        pbar = tqdm(loader, desc=f"Stage{stage} Epoch {epoch}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)

            with torch.no_grad():
                z_ir = encode_vae(vae_ir, ir)
                z_vis = encode_vae(vae_vis, vis)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if stage == 1:
                    outputs = model.forward_stage1(z_ir, z_vis)

                    ir_rec = decode_vae(vae_ir, outputs["z_ir_rec"])
                    vis_rec = decode_vae(vae_vis, outputs["z_vis_rec"])

                    l_rec_ir = F.l1_loss(ir_rec, ir)
                    l_rec_vis = F.l1_loss(vis_rec, vis)

                    l_decomp, l_cdc, l_cbc = decomposition_loss(
                        outputs["phi_id"],
                        outputs["phi_vd"],
                        outputs["phi_ib"],
                        outputs["phi_vb"],
                        cfg.epsilon_decomp,
                    )

                    loss = (
                        cfg.recon_weight_ir * l_rec_ir
                        + cfg.recon_weight_vis * l_rec_vis
                        + cfg.decomp_weight_stage1 * l_decomp
                    )

                    l_int = torch.zeros_like(loss)
                    l_grad = torch.zeros_like(loss)
                else:
                    outputs = model.forward_stage2(z_ir, z_vis)
                    fused = decode_vae(vae_vis, outputs["z_fused"])

                    l_int = 0.5 * F.l1_loss(fused, vis) + 0.5 * F.l1_loss(
                        rgb_to_gray(fused), rgb_to_gray(ir)
                    )

                    grad_fused = sobel_magnitude(fused)
                    grad_target = torch.maximum(
                        sobel_magnitude(vis),
                        sobel_magnitude(ir),
                    )
                    l_grad = F.l1_loss(grad_fused, grad_target)

                    l_decomp, l_cdc, l_cbc = decomposition_loss(
                        outputs["phi_id"],
                        outputs["phi_vd"],
                        outputs["phi_ib"],
                        outputs["phi_vb"],
                        cfg.epsilon_decomp,
                    )

                    loss = (
                        cfg.intensity_weight * l_int
                        + cfg.grad_weight * l_grad
                        + cfg.decomp_weight_stage2 * l_decomp
                    )

                    l_rec_ir = torch.zeros_like(loss)
                    l_rec_vis = torch.zeros_like(loss)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            stats["loss"] += float(loss.item())
            stats["l_rec_ir"] += float(l_rec_ir.item())
            stats["l_rec_vis"] += float(l_rec_vis.item())
            stats["l_int"] += float(l_int.item())
            stats["l_grad"] += float(l_grad.item())
            stats["l_decomp"] += float(l_decomp.item())
            stats["l_cdc"] += float(l_cdc.item())
            stats["l_cbc"] += float(l_cbc.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                decomp=f"{l_decomp.item():.4f}",
                cdc=f"{l_cdc.item():.4f}",
                cbc=f"{l_cbc.item():.4f}",
            )

            if batch_idx == 0 and (epoch % cfg.save_every == 0):
                if stage == 1:
                    with torch.no_grad():
                        vis_preview = decode_vae(vae_vis, outputs["z_vis_rec"])
                    save_visualization(output_dir / "samples" / "stage1", f"e{epoch:03d}", ir, vis, vis_preview)
                else:
                    save_visualization(output_dir / "samples" / "stage2", f"e{epoch:03d}", ir, vis, fused)

        n = len(loader)
        for k in stats:
            stats[k] /= max(n, 1)
        return stats

    # Stage 1: decomposition + reconstruction warm-up.
    # Train align/decompose/reconstruct modules first for better stage-2 stability.
    LOGGER.info("Starting Stage 1 training for %d epochs", cfg.epochs_stage1)
    for epoch in range(1, cfg.epochs_stage1 + 1):
        s = run_epoch(epoch, stage=1)
        LOGGER.info(
            "[Stage1][Epoch %03d] loss=%.6f rec_ir=%.6f rec_vis=%.6f decomp=%.6f cdc=%.6f cbc=%.6f",
            epoch,
            s["loss"],
            s["l_rec_ir"],
            s["l_rec_vis"],
            s["l_decomp"],
            s["l_cdc"],
            s["l_cbc"],
        )

        ckpt = {
            "epoch": epoch,
            "stage": 1,
            "model": model.state_dict(),
            "optimizer": opt_stage1.state_dict(),
            "config": asdict(cfg),
        }
        torch.save(ckpt, output_dir / f"stage1_epoch_{epoch:03d}.pt")

    # Stage 2: fusion training.
    LOGGER.info("Starting Stage 2 training for %d epochs", cfg.epochs_stage2)
    for epoch in range(1, cfg.epochs_stage2 + 1):
        s = run_epoch(epoch, stage=2)
        LOGGER.info(
            "[Stage2][Epoch %03d] loss=%.6f int=%.6f grad=%.6f decomp=%.6f cdc=%.6f cbc=%.6f",
            epoch,
            s["loss"],
            s["l_int"],
            s["l_grad"],
            s["l_decomp"],
            s["l_cdc"],
            s["l_cbc"],
        )

        ckpt = {
            "epoch": epoch,
            "stage": 2,
            "model": model.state_dict(),
            "optimizer": opt_stage2.state_dict(),
            "config": asdict(cfg),
        }
        torch.save(ckpt, output_dir / f"stage2_epoch_{epoch:03d}.pt")

    torch.save(model.state_dict(), output_dir / "dual_vae_fusion_final.pt")
    LOGGER.info("Training completed. Final model saved to %s", output_dir / "dual_vae_fusion_final.pt")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Dual-VAE infrared/visible fusion training")

    parser.add_argument("--sd_ckpt_path", type=str, default="./model/v1-5-pruned.ckpt")
    parser.add_argument("--ir_vae_path", type=str, default="./sd15_ir_vae_512_10k_lpips/vae_best_lpips")
    parser.add_argument("--ir_data_dir", type=str, default="/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Ir")
    parser.add_argument("--vis_data_dir", type=str, default="/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Vis")
    parser.add_argument("--output_dir", type=str, default="./dual_vae_fusion_runs")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr_stage1", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--epochs_stage1", type=int, default=10)
    parser.add_argument("--epochs_stage2", type=int, default=30)

    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--feature_channels", type=int, default=64)

    parser.add_argument("--recon_weight_ir", type=float, default=1.0)
    parser.add_argument("--recon_weight_vis", type=float, default=1.0)
    parser.add_argument("--decomp_weight_stage1", type=float, default=0.5)

    parser.add_argument("--intensity_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.2)
    parser.add_argument("--decomp_weight_stage2", type=float, default=0.2)
    parser.add_argument("--epsilon_decomp", type=float, default=1.01)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    return TrainConfig(
        sd_ckpt_path=args.sd_ckpt_path,
        ir_vae_path=args.ir_vae_path,
        ir_data_dir=args.ir_data_dir,
        vis_data_dir=args.vis_data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        weight_decay=args.weight_decay,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        latent_channels=args.latent_channels,
        feature_channels=args.feature_channels,
        recon_weight_ir=args.recon_weight_ir,
        recon_weight_vis=args.recon_weight_vis,
        decomp_weight_stage1=args.decomp_weight_stage1,
        intensity_weight=args.intensity_weight,
        grad_weight=args.grad_weight,
        decomp_weight_stage2=args.decomp_weight_stage2,
        epsilon_decomp=args.epsilon_decomp,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        seed=args.seed,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
