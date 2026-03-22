import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .color import colorize_with_vis_chroma, rgb_to_gray
from .configs import TrainConfig
from .datasets import PairedInfraredVisibleDataset
from .losses import decomposition_loss, make_ir_saliency_mask, saliency_masked_l1, sobel_magnitude, ssim_loss
from .models import DualVAEFusionModel
from .vae_utils import decode_vae, encode_vae, load_sd_vae, maybe_validate_vis_vae

LOGGER = logging.getLogger("dual_vae_fusion")

STAGE1_KEY_PREFIXES = (
    "align_ir.",
    "align_vis.",
    "decomp_ir.",
    "decomp_vis.",
    "rec_ir.",
    "rec_vis.",
)

ALLOWED_RESUME_MISSING_PREFIXES = (
    "decoder_adapter.attn_q_heads.",
    "decoder_adapter.attn_k_heads.",
    "decoder_adapter.attn_v_heads.",
    "decoder_adapter.attn_out_heads.",
    "decoder_adapter.attn_gamma",
)


def laplacian_detail(x: torch.Tensor) -> torch.Tensor:
    return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "train.log", encoding="utf-8")],
    )


def load_training_checkpoint(path: str, model: nn.Module, start_stage: int) -> Dict[str, object]:
    LOGGER.info("Loading resume checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint format: {path}")
    ckpt_stage = int(ckpt.get("stage", 0) or 0)

    # Keep Stage 1 training independent of Stage 2 architecture changes.
    if start_stage == 1 and ckpt_stage == 1:
        stage1_sd = {k: v for k, v in ckpt["model"].items() if k.startswith(STAGE1_KEY_PREFIXES)}
        missing, unexpected = model.load_state_dict(stage1_sd, strict=False)
        missing_stage1 = [k for k in missing if k.startswith(STAGE1_KEY_PREFIXES)]
        if missing_stage1 or unexpected:
            raise RuntimeError(
                f"Stage1 checkpoint load mismatch. missing_stage1={missing_stage1}, unexpected={unexpected}"
            )
        LOGGER.info("Loaded Stage1 submodules from checkpoint (partial load). keys=%d", len(stage1_sd))
    else:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        unexpected_filtered = [k for k in unexpected if k]
        disallowed_missing = [
            k for k in missing if not any(k.startswith(prefix) for prefix in ALLOWED_RESUME_MISSING_PREFIXES)
        ]
        if disallowed_missing or unexpected_filtered:
            raise RuntimeError(
                f"Checkpoint load mismatch. missing={disallowed_missing}, unexpected={unexpected_filtered}"
            )
    LOGGER.info("Checkpoint loaded. stage=%s epoch=%s", ckpt.get("stage"), ckpt.get("epoch"))
    return ckpt


def save_visualization(save_dir: Path, step_tag: str, ir: torch.Tensor, vis: torch.Tensor, fused: torch.Tensor) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    panel = torch.cat([ir[:1], vis[:1], fused[:1]], dim=0)
    panel = (panel.clamp(-1, 1) + 1.0) / 2.0
    save_image(panel, save_dir / f"{step_tag}.png", nrow=3)


def train(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    setup_logging(output_dir / "logs")
    LOGGER.info("Training configuration: %s", json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    vis_ratio = max(float(cfg.stage2_vis_recon_ratio), 0.0)
    ir_ratio = max(float(cfg.stage2_ir_gray_ratio), 0.0)
    ir_sal_ratio = max(float(cfg.stage2_ir_salient_ratio), 0.0)
    ratio_sum = vis_ratio + ir_ratio + ir_sal_ratio
    if ratio_sum <= 0:
        raise RuntimeError("Invalid stage2 ratios: vis/ir_gray/ir_salient all <=0")
    vis_ratio, ir_ratio, ir_sal_ratio = vis_ratio / ratio_sum, ir_ratio / ratio_sum, ir_sal_ratio / ratio_sum

    LOGGER.info("Loading SD1.5 RGB VAE...")
    vae_vis = load_sd_vae(cfg, LOGGER, device)
    maybe_validate_vis_vae(vae_vis, cfg, LOGGER, device, str(output_dir / "vis_roundtrip"))

    LOGGER.info("Loading IR VAE from directory: %s", cfg.ir_vae_path)
    vae_ir = AutoencoderKL.from_pretrained(cfg.ir_vae_path, torch_dtype=torch.float32, local_files_only=True).to(device)
    vae_ir.eval().requires_grad_(False)
    vae_vis.eval().requires_grad_(False)

    if not cfg.force_resize and cfg.batch_size != 1:
        raise RuntimeError("When --no_force_resize is enabled, set --batch_size 1 to avoid variable-shape collation errors.")

    dataset = PairedInfraredVisibleDataset(
        cfg.ir_data_dir,    
        cfg.vis_data_dir,
        cfg.image_size,
        force_resize=cfg.force_resize,
        logger=LOGGER,
    )
    cpu_count = os.cpu_count() or 1
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=min(cfg.num_workers, max(0, cpu_count - 1)), pin_memory=(device.type == "cuda"), drop_last=True)

    model = DualVAEFusionModel(cfg.latent_channels, cfg.feature_channels).to(device)
    stage1_params = list(model.align_ir.parameters()) + list(model.align_vis.parameters()) + list(model.decomp_ir.parameters()) + list(model.decomp_vis.parameters()) + list(model.rec_ir.parameters()) + list(model.rec_vis.parameters())
    opt_stage1 = torch.optim.AdamW(stage1_params, lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
    opt_stage2 = torch.optim.AdamW(model.parameters(), lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)

    resume_meta: Dict[str, object] = {}
    if cfg.resume_ckpt:
        resume_meta = load_training_checkpoint(cfg.resume_ckpt, model, cfg.start_stage)

    def run_epoch(epoch: int, stage: int) -> Dict[str, float]:
        model.train()
        optimizer = opt_stage1 if stage == 1 else opt_stage2
        stats = {k: 0.0 for k in ["loss", "l_int", "l_in_max", "l_grad_max", "l_vis_bg_int", "l_color_sal", "l_color_bg", "l_decomp", "l_ir_edge", "l_vis_edge_bg", "l_vis_hf", "l_vis_hf_bg", "l_vis_hf_sal", "l_vis_lap", "l_vis_lap_bg", "l_vis_lap_sal", "l_edge_focus", "l_ssim", "l_saliency"]}
        pbar = tqdm(loader, desc=f"Stage{stage} Epoch {epoch}", leave=False)
        running_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            z_ir, z_vis = encode_vae(vae_ir, ir), encode_vae(vae_vis, vis)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if stage == 1:
                    out = model.forward_stage1(z_ir, z_vis)
                    ir_rec = decode_vae(vae_ir, out["z_ir_rec"])
                    vis_rec = decode_vae(vae_vis, out["z_vis_rec"])
                    l_decomp, *_ = decomposition_loss(out["phi_id"], out["phi_vd"], out["phi_ib"], out["phi_vb"], cfg.epsilon_decomp)
                    loss = cfg.recon_weight_ir * F.l1_loss(ir_rec, ir) + cfg.recon_weight_vis * F.l1_loss(vis_rec, vis) + cfg.decomp_weight_stage1 * l_decomp
                    fused = vis_rec
                    l_int = l_in_max = l_grad_max = l_vis_bg_int = l_color_sal = l_color_bg = torch.zeros_like(loss)
                    l_ir_edge = l_vis_edge_bg = l_vis_hf = l_vis_hf_bg = l_vis_hf_sal = l_vis_lap = l_vis_lap_bg = l_vis_lap_sal = l_edge_focus = l_ssim = torch.zeros_like(loss)
                    l_saliency = torch.zeros_like(loss)
                    # l_haze
                else:
                    out = model.forward_stage2(z_ir, z_vis)
                    fused_raw = decode_vae(
                        vae_vis,
                        out["z_fused"],
                        decoder_adapter=model.decoder_adapter,
                        adapter_cond=out.get("decoder_adapter_cond"),
                    )
                    gray_fused = rgb_to_gray(fused_raw)
                    if cfg.output_gray_only:
                        fused = gray_fused.repeat(1, 3, 1, 1)
                        vis_target = rgb_to_gray(vis).repeat(1, 3, 1, 1)
                    else:
                        fused = colorize_with_vis_chroma(gray_fused, vis)
                        vis_target = vis
                    gray_ir, y_vis = rgb_to_gray(ir), rgb_to_gray(vis)
                    ir_mask = make_ir_saliency_mask(gray_ir, cfg.ir_saliency_tau, cfg.ir_saliency_temp)
                    l_int_vis = F.l1_loss(fused, vis_target)
                    l_int_ir_global = F.l1_loss(gray_fused, gray_ir)
                    l_int_ir_salient = (ir_mask * (gray_fused - gray_ir).abs()).sum() / (ir_mask.sum() + 1e-6)
                    y_max = torch.maximum(y_vis, gray_ir)
                    saliency_target = cfg.saliency_ir_target_ratio * gray_ir + (1.0 - cfg.saliency_ir_target_ratio) * y_vis
                    l_saliency = saliency_masked_l1(gray_fused, saliency_target, ir_mask)
                    l_int = vis_ratio * l_int_vis + ir_ratio * l_int_ir_global + ir_sal_ratio * l_int_ir_salient
                    bg_mask = 1.0 - ir_mask
                    l_vis_bg_int = (bg_mask * (gray_fused - y_vis).abs()).sum() / (bg_mask.sum() + 1e-6)

                    l_in_max = F.l1_loss(gray_fused, y_max)
                    sobel_fused = sobel_magnitude(gray_fused)
                    sobel_ir = sobel_magnitude(gray_ir)
                    sobel_vis = sobel_magnitude(y_vis)
                    edge_ref = torch.maximum(sobel_vis, sobel_ir)

                    l_grad_max = F.l1_loss(sobel_fused, edge_ref)
                    l_color_sal = (ir_mask * (fused - vis).abs()).sum() / (ir_mask.sum() * fused.shape[1] + 1e-6)
                    l_color_bg = (bg_mask * (fused - vis).abs()).sum() / (bg_mask.sum() * fused.shape[1] + 1e-6)

                    # Encourage thermal structures to survive fusion in salient IR regions.
                    l_ir_edge = (ir_mask * (sobel_fused - sobel_ir).abs()).sum() / (ir_mask.sum() + 1e-6)
                    l_vis_edge_bg = (bg_mask * (sobel_fused - sobel_vis).abs()).sum() / (bg_mask.sum() + 1e-6)
                    l_vis_hf = F.l1_loss(sobel_fused, sobel_vis)
                    l_vis_hf_bg = (bg_mask * (sobel_fused - sobel_vis).abs()).sum() / (bg_mask.sum() + 1e-6)
                    l_vis_hf_sal = (ir_mask * (sobel_fused - sobel_vis).abs()).sum() / (ir_mask.sum() + 1e-6)
                    lap_fused = laplacian_detail(gray_fused)
                    lap_vis = laplacian_detail(y_vis)
                    l_vis_lap = F.l1_loss(lap_fused, lap_vis)
                    l_vis_lap_bg = (bg_mask * (lap_fused - lap_vis).abs()).sum() / (bg_mask.sum() + 1e-6)
                    l_vis_lap_sal = (ir_mask * (lap_fused - lap_vis).abs()).sum() / (ir_mask.sum() + 1e-6)

                    # Focus sharpening on strong edges to avoid over-smoothing fine structures.
                    edge_focus = (edge_ref / (edge_ref.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)).clamp(0.5, 3.0).detach()
                    l_edge_focus = (edge_focus * (sobel_fused - edge_ref).abs()).mean()

                    l_decomp, *_ = decomposition_loss(out["phi_id"], out["phi_vd"], out["phi_ib"], out["phi_vb"], cfg.epsilon_decomp)
                    l_ssim = ssim_loss(gray_fused, y_max)

                    # grad_vis = sobel_magnitude(y_vis)
                    # haze_mask = make_haze_mask(y_vis, grad_vis, cfg.haze_brightness_tau, cfg.haze_grad_tau, cfg.haze_mask_temp)
                    # l_haze_int = masked_l1(gray_fused, y_max, haze_mask)
                    # l_haze_grad = masked_l1(sobel_magnitude(gray_fused), sobel_magnitude(y_max), haze_mask)
                    # l_haze = l_haze_int + 0.5 * l_haze_grad

                    loss = (
                        cfg.fusion_max_int_weight * l_in_max
                        + cfg.fusion_max_grad_weight * l_grad_max
                        + cfg.stage2_ssim_weight * l_ssim
                        + cfg.saliency_weight * l_saliency
                        + cfg.intensity_weight * l_int
                        + cfg.stage2_vis_bg_int_weight * l_vis_bg_int
                        + (cfg.fusion_color_sal_weight * cfg.stage2_salient_color_suppress) * l_color_sal
                        + cfg.fusion_color_bg_weight * l_color_bg
                        + cfg.stage2_ir_edge_weight * l_ir_edge
                        + cfg.stage2_vis_edge_bg_weight * l_vis_edge_bg
                        + cfg.stage2_vis_hf_weight * l_vis_hf
                        + cfg.stage2_vis_hf_bg_weight * l_vis_hf_bg
                        + cfg.stage2_vis_hf_sal_weight * l_vis_hf_sal
                        + cfg.stage2_vis_lap_weight * l_vis_lap
                        + cfg.stage2_vis_lap_bg_weight * l_vis_lap_bg
                        + cfg.stage2_vis_lap_sal_weight * l_vis_lap_sal
                        + cfg.stage2_edge_focus_weight * l_edge_focus
                        + cfg.decomp_weight_stage2 * l_decomp
                    )
                    # + cfg.haze_loss_weight * l_haze

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for k, v in {"loss": loss, "l_int": l_int, "l_in_max": l_in_max, "l_grad_max": l_grad_max, "l_vis_bg_int": l_vis_bg_int, "l_color_sal": l_color_sal, "l_color_bg": l_color_bg, "l_decomp": l_decomp, "l_ir_edge": l_ir_edge, "l_vis_edge_bg": l_vis_edge_bg, "l_vis_hf": l_vis_hf, "l_vis_hf_bg": l_vis_hf_bg, "l_vis_hf_sal": l_vis_hf_sal, "l_vis_lap": l_vis_lap, "l_vis_lap_bg": l_vis_lap_bg, "l_vis_lap_sal": l_vis_lap_sal, "l_edge_focus": l_edge_focus, "l_ssim": l_ssim, "l_saliency": l_saliency}.items():
                stats[k] += float(v.item())

            running_loss += float(loss.item())
            avg_loss_so_far = running_loss / float(batch_idx + 1)
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", avg_loss=f"{avg_loss_so_far:.4f}")

            if batch_idx == 0 and (epoch % cfg.save_every == 0):
                save_visualization(output_dir / "samples" / f"stage{stage}", f"e{epoch:03d}", ir, vis, fused)

        n = len(loader)
        return {k: v / max(n, 1) for k, v in stats.items()}

    if cfg.start_stage <= 1 and cfg.epochs_stage1 > 0:
        start = int(resume_meta.get("epoch", 0)) + 1 if cfg.resume_ckpt and int(resume_meta.get("stage", 0)) == 1 else 1
        for epoch in range(start, cfg.epochs_stage1 + 1):
            s = run_epoch(epoch, stage=1)
            LOGGER.info("[Stage1][Epoch %03d] loss=%.6f decomp=%.6f", epoch, s["loss"], s["l_decomp"])
            torch.save({"epoch": epoch, "stage": 1, "model": model.state_dict(), "optimizer": opt_stage1.state_dict(), "config": asdict(cfg)}, output_dir / f"stage1_epoch_{epoch:03d}.pt")

    if cfg.epochs_stage2 > 0 and cfg.start_stage <= 2:
        LOGGER.info("Preparing for Stage 2: Freezing VAE components and optimizing Fusion Adapter...")
        
        # Lock all parameters first, then selectively unfreeze modules for Stage 2.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fusion.parameters():
            p.requires_grad = True
        for p in model.decoder_adapter.parameters():
            p.requires_grad = True
        if cfg.stage2_unfreeze_ir_branches:
            for p in model.align_ir.parameters():
                p.requires_grad = True
            for p in model.decomp_ir.parameters():
                p.requires_grad = True
        if cfg.stage2_unfreeze_vis_align:
            for p in model.align_vis.parameters():
                p.requires_grad = True
        
        # 重新定义优化器，只更新 fusion 模块
        opt_stage2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr_stage2)

        best_loss = float("inf")
        best_epoch = -1
        stage2_loss_sum = 0.0
        stage2_epoch_count = 0
        
        start = int(resume_meta.get("epoch", 0)) + 1 if cfg.resume_ckpt and int(resume_meta.get("stage", 0)) == 2 else 1
        end_epoch = int(cfg.epochs_stage2)
        if start > 1 and end_epoch < start:
            # If resuming Stage2 from a later epoch and user sets a smaller number,
            # treat epochs_stage2 as "additional epochs to run".
            end_epoch = start - 1 + max(int(cfg.epochs_stage2), 0)
            LOGGER.info(
                "Resumed Stage2 at epoch=%d with epochs_stage2=%d. Interpreting as additional epochs; new end_epoch=%d",
                start - 1,
                int(cfg.epochs_stage2),
                end_epoch,
            )
        if end_epoch < start:
            LOGGER.warning(
                "Stage2 loop skipped: start_epoch=%d, end_epoch=%d. Increase epochs_stage2 or disable resume.",
                start,
                end_epoch,
            )

        for epoch in range(start, end_epoch + 1):
            s = run_epoch(epoch, stage=2)
            # s["loss"] is already mean loss over all batches in this epoch.
            epoch_avg_loss = float(s["loss"])
            stage2_loss_sum += epoch_avg_loss
            stage2_epoch_count += 1
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": 2,
                        "best_epoch": best_epoch,
                        "best_loss": best_loss,
                        "model": model.state_dict(),
                        "optimizer": opt_stage2.state_dict(),
                        "config": asdict(cfg),
                    },
                    output_dir / "stage2_best.pt",
                )
            LOGGER.info("[Stage2][Epoch %03d] loss=%.6f in_max=%.6f grad_max=%.6f int=%.6f saliency=%.6f vis_bg_int=%.6f c_sal=%.6f c_bg=%.6f ir_edge=%.6f vis_edge_bg=%.6f vis_hf=%.6f vis_hf_bg=%.6f vis_hf_sal=%.6f vis_lap=%.6f vis_lap_bg=%.6f vis_lap_sal=%.6f edge_focus=%.6f ssim=%.6f", epoch, s["loss"], s["l_in_max"], s["l_grad_max"], s["l_int"], s["l_saliency"], s["l_vis_bg_int"], s["l_color_sal"], s["l_color_bg"], s["l_ir_edge"], s["l_vis_edge_bg"], s["l_vis_hf"], s["l_vis_hf_bg"], s["l_vis_hf_sal"], s["l_vis_lap"], s["l_vis_lap_bg"], s["l_vis_lap_sal"], s["l_edge_focus"], s["l_ssim"])
            LOGGER.info("[Stage2][Epoch %03d] average_loss=%.6f best_loss=%.6f best_epoch=%d", epoch, epoch_avg_loss, best_loss, best_epoch)
            torch.save({"epoch": epoch, "stage": 2, "best_epoch": best_epoch, "best_loss": best_loss, "model": model.state_dict(), "optimizer": opt_stage2.state_dict(), "config": asdict(cfg)}, output_dir / f"stage2_epoch_{epoch:03d}.pt")

        if stage2_epoch_count > 0:
            average_loss = stage2_loss_sum / float(stage2_epoch_count)
            LOGGER.info("[Stage2] training average_loss=%.6f across %d epochs", average_loss, stage2_epoch_count)
            LOGGER.info("[Stage2] best_loss=%.6f at epoch=%d", best_loss, best_epoch)

    torch.save(model.state_dict(), output_dir / "dual_vae_fusion_final.pt")
    LOGGER.info("Training completed. Final model saved to %s", output_dir / "dual_vae_fusion_final.pt")
