from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ir_sr_project.datasets.ir_sr_dataset import DatasetConfig, IRSRDataset
from ir_sr_project.models.corple_student import CoRPLELite
from ir_sr_project.models.losses import DistillLoss, LPIPSLoss, freq_loss, l1_loss
from ir_sr_project.models.prompt_guidance import CLIPPromptLoss, PromptConfig
from ir_sr_project.models.teacher_adapter import TeacherAdapter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = torch.mean((sr - hr) ** 2).item() + 1e-12
    return float(10.0 * np.log10(1.0 / mse))


def build_loader(cfg: dict, split: str, batch_size: int, workers: int) -> DataLoader:
    dcfg = DatasetConfig(
        root=cfg["root"],
        split=cfg[f"{split}_split"],
        hr_subdir=cfg["hr_subdir"],
        lr_subdir=cfg.get("lr_subdir", "LR"),
        use_precomputed_lr=bool(cfg.get("use_precomputed_lr", False)),
        exts=tuple(cfg["exts"]),
        scale=cfg["scale"],
        hr_size=cfg["hr_size"],
    )
    ds = IRSRDataset(dcfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=workers, pin_memory=True)


def resolve_ckpt_names(stage_name: str) -> tuple[str, str]:
    if stage_name:
        return f"last_{stage_name}.pth", f"best_{stage_name}.pth"
    return "last.pth", "best.pth"


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    resume_path: str,
    resume_mode: str,
    reset_best_on_resume: bool,
) -> tuple[int, float]:
    """Return (start_epoch, best_psnr)."""
    if not resume_path:
        return 1, -1.0

    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"resume_ckpt not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    start_epoch = 1
    best_psnr = -1.0

    if resume_mode == "continue":
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if not reset_best_on_resume:
            best_psnr = float(ckpt.get("best_psnr", ckpt.get("psnr", -1.0)))

    print(
        f"[Resume] mode={resume_mode}, path={path}, start_epoch={start_epoch}, "
        f"best_psnr_init={best_psnr:.4f}"
    )
    return start_epoch, best_psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_name = str(cfg["train"].get("stage_name", "")).strip()
    last_name, best_name = resolve_ckpt_names(stage_name)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(cfg["data"], "train", cfg["train"]["batch_size"], cfg["data"]["num_workers"])
    val_loader = build_loader(cfg["data"], "val", 1, cfg["data"]["num_workers"])
    print(f"[Data] train={len(train_loader.dataset)} images, val={len(val_loader.dataset)} images")

    model = CoRPLELite(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"])
    )
    scaler = GradScaler(enabled=bool(cfg["runtime"].get("amp", True)))

    start_epoch, best_psnr = maybe_resume(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=str(cfg["train"].get("resume_ckpt", "")).strip(),
        resume_mode=str(cfg["train"].get("resume_mode", "finetune")).strip(),
        reset_best_on_resume=bool(cfg["train"].get("reset_best_on_resume", True)),
    )

    lpips_loss = LPIPSLoss().to(device)

    use_distill = bool(cfg["train"].get("use_teacher_distill", False))
    teacher = None
    distill_loss_fn = None
    if use_distill:
        teacher = TeacherAdapter(
            teacher_path=cfg["train"]["teacher_path"],
            layer_keys=cfg["train"].get("teacher_layer_keys", ["down0", "mid", "up0"]),
        ).to(device)
        distill_loss_fn = DistillLoss(
            stu_channels=[cfg["model"]["feat_ch"]] * 3,
            tea_channels=[1, 1, 1],
            proj_ch=cfg["model"]["feat_ch"],
        ).to(device)

    use_prompt_loss = bool(cfg["train"].get("use_prompt_loss", False))
    prompt_loss_fn = None
    if use_prompt_loss:
        pcfg = cfg.get("prompt", {})
        prompt_loss_fn = CLIPPromptLoss(
            PromptConfig(
                model_name_or_path=pcfg.get("model_name_or_path", "model/clip-vit-large-patch14"),
                positive_prompt=pcfg.get("positive_prompt", "a high quality infrared image with clear thermal edges"),
                negative_prompt=pcfg.get("negative_prompt", "a blurry noisy low quality infrared image"),
                margin=float(pcfg.get("margin", 0.1)),
            )
        ).to(device)

    epochs = int(cfg["train"]["epochs"])

    for ep in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=bool(cfg["runtime"].get("amp", True))):
                sr, stu_feats = model(lr, return_feats=True)

                loss_l1 = l1_loss(sr, hr)
                loss_lp = lpips_loss(sr, hr)
                loss_fr = freq_loss(sr, hr)

                loss = (
                    cfg["loss"]["w_l1"] * loss_l1
                    + cfg["loss"]["w_lpips"] * loss_lp
                    + cfg["loss"]["w_freq"] * loss_fr
                )

                loss_dis = torch.tensor(0.0, device=device)
                if use_distill and teacher is not None and distill_loss_fn is not None:
                    tea_feats = teacher.extract(lr)
                    loss_dis = distill_loss_fn(stu_feats, tea_feats)
                    loss = loss + cfg["loss"]["w_distill"] * loss_dis

                loss_prompt = torch.tensor(0.0, device=device)
                prompt_stats = {"s_pos": 0.0, "s_neg": 0.0}
                if use_prompt_loss and prompt_loss_fn is not None:
                    with autocast(enabled=False):
                        loss_prompt, prompt_stats = prompt_loss_fn(sr.float())
                    loss = loss + float(cfg["loss"].get("w_prompt", 0.02)) * loss_prompt

            scaler.scale(loss).backward()
            if float(cfg["train"]["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(
                l1=float(loss_l1.item()),
                lpips=float(loss_lp.item()),
                freq=float(loss_fr.item()),
                distill=float(loss_dis.item()),
                prompt=float(loss_prompt.item()),
                s_pos=prompt_stats["s_pos"],
                s_neg=prompt_stats["s_neg"],
            )

        if ep % int(cfg["train"]["val_every"]) == 0:
            model.eval()
            scores = []
            with torch.no_grad():
                for batch in val_loader:
                    lr = batch["lr"].to(device)
                    hr = batch["hr"].to(device)
                    sr = model(lr)
                    scores.append(psnr(sr, hr))
            val_psnr = float(np.mean(scores)) if len(scores) > 0 else 0.0
            print(f"[Val] epoch={ep} PSNR={val_psnr:.3f}")

            ckpt_last = out_dir / last_name
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "psnr": val_psnr,
                    "best_psnr": best_psnr,
                    "config": cfg,
                },
                ckpt_last,
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {
                        "epoch": ep,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "psnr": val_psnr,
                        "best_psnr": best_psnr,
                        "config": cfg,
                    },
                    out_dir / best_name,
                )

        if ep % int(cfg["train"]["save_every"]) == 0:
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_psnr": best_psnr,
                    "config": cfg,
                },
                out_dir / f"epoch_{ep}.pth",
            )

    print(f"Training complete. Best PSNR={best_psnr:.3f}")


if __name__ == "__main__":
    main()
