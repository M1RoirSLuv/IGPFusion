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
        exts=tuple(cfg["exts"]),
        scale=cfg["scale"],
        hr_size=cfg["hr_size"],
    )
    ds = IRSRDataset(dcfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(cfg["data"], "train", cfg["train"]["batch_size"], cfg["data"]["num_workers"])
    val_loader = build_loader(cfg["data"], "val", 1, cfg["data"]["num_workers"])

    model = CoRPLELite(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"])
    )
    scaler = GradScaler(enabled=bool(cfg["runtime"].get("amp", True)))

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

    best_psnr = -1.0
    epochs = int(cfg["train"]["epochs"])

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=bool(cfg["runtime"].get("amp", True))):
                sr, stu_feats = model(lr, return_feats=True)

                loss = 0.0
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

            ckpt_last = out_dir / "last.pth"
            torch.save({"epoch": ep, "model": model.state_dict(), "psnr": val_psnr, "config": cfg}, ckpt_last)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({"epoch": ep, "model": model.state_dict(), "psnr": val_psnr, "config": cfg}, out_dir / "best.pth")

        if ep % int(cfg["train"]["save_every"]) == 0:
            torch.save({"epoch": ep, "model": model.state_dict(), "config": cfg}, out_dir / f"epoch_{ep}.pth")

    print(f"Training complete. Best PSNR={best_psnr:.3f}")


if __name__ == "__main__":
    main()
