from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class DatasetConfig:
    root: str
    split: str
    hr_subdir: str = "HR"
    lr_subdir: str = "LR"
    use_precomputed_lr: bool = False
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")
    scale: int = 4
    hr_size: int = 256


class IRSRDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.hr_paths = self._scan_paths(cfg.hr_subdir)
        if len(self.hr_paths) == 0:
            raise FileNotFoundError(f"No HR images found in {cfg.root}/{cfg.split}/{cfg.hr_subdir}")

        self.lr_by_stem: Dict[str, Path] = {}
        if self.cfg.use_precomputed_lr:
            lr_paths = self._scan_paths(cfg.lr_subdir)
            if len(lr_paths) == 0:
                raise FileNotFoundError(f"No LR images found in {cfg.root}/{cfg.split}/{cfg.lr_subdir}")
            self.lr_by_stem = {p.stem: p for p in lr_paths}
            missing = [p.name for p in self.hr_paths if p.stem not in self.lr_by_stem]
            if missing:
                sample = ", ".join(missing[:5])
                raise FileNotFoundError(
                    "Precomputed LR mode requires filename stem alignment between HR/LR. "
                    f"Missing LR for {len(missing)} HR files. Example: {sample}"
                )

    def _scan_paths(self, subdir: str) -> List[Path]:
        root = Path(self.cfg.root) / self.cfg.split / subdir
        paths: List[Path] = []
        for ext in self.cfg.exts:
            paths.extend(sorted(root.glob(f"*.{ext}")))
            paths.extend(sorted(root.glob(f"*.{ext.upper()}")))
        return sorted(set(paths))

    def _read_gray(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        return TF.to_tensor(img)  # [1,H,W], 0..1

    def _random_crop_pair(self, lr: torch.Tensor, hr: torch.Tensor, hr_size: int, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_hr, w_hr = hr.shape
        hr_size = min(hr_size, h_hr - (h_hr % scale), w_hr - (w_hr % scale))
        if hr_size <= 0:
            raise ValueError("Invalid HR size for crop in precomputed LR mode.")
        lr_size = hr_size // scale

        _, h_lr, w_lr = lr.shape
        if h_lr < lr_size or w_lr < lr_size:
            raise ValueError(
                f"LR image is too small for crop size. LR shape=({h_lr},{w_lr}), needed={lr_size}."
            )

        top_lr = torch.randint(0, h_lr - lr_size + 1, (1,)).item()
        left_lr = torch.randint(0, w_lr - lr_size + 1, (1,)).item()
        top_hr, left_hr = top_lr * scale, left_lr * scale

        lr_crop = lr[:, top_lr : top_lr + lr_size, left_lr : left_lr + lr_size]
        hr_crop = hr[:, top_hr : top_hr + hr_size, left_hr : left_hr + hr_size]
        return lr_crop, hr_crop

    def _random_crop(self, x: torch.Tensor, size: int) -> torch.Tensor:
        _, h, w = x.shape
        if h < size or w < size:
            x = TF.resize(x, [max(h, size), max(w, size)], antialias=True)
            _, h, w = x.shape
        top = torch.randint(0, h - size + 1, (1,)).item()
        left = torch.randint(0, w - size + 1, (1,)).item()
        return x[:, top : top + size, left : left + size]

    def _to_lr(self, hr: torch.Tensor) -> torch.Tensor:
        s = self.cfg.scale
        h, w = hr.shape[-2:]
        lr = F.interpolate(hr.unsqueeze(0), size=(h // s, w // s), mode="bicubic", align_corners=False)
        return lr.squeeze(0)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr_path = self.hr_paths[idx]
        hr = self._read_gray(hr_path)

        if self.cfg.use_precomputed_lr:
            lr = self._read_gray(self.lr_by_stem[hr_path.stem])
            if self.cfg.split == "train":
                lr, hr = self._random_crop_pair(lr, hr, self.cfg.hr_size, self.cfg.scale)
                if torch.rand(1).item() < 0.5:
                    hr = TF.hflip(hr)
                    lr = TF.hflip(lr)
                if torch.rand(1).item() < 0.5:
                    hr = TF.vflip(hr)
                    lr = TF.vflip(lr)
            else:
                s = self.cfg.scale
                h = min(hr.shape[-2], lr.shape[-2] * s)
                w = min(hr.shape[-1], lr.shape[-1] * s)
                h = h - (h % s)
                w = w - (w % s)
                hr = hr[:, :h, :w]
                lr = lr[:, : h // s, : w // s]
            return {"lr": lr, "hr": hr, "name": hr_path.stem}

        if self.cfg.split == "train":
            hr = self._random_crop(hr, self.cfg.hr_size)
            if torch.rand(1).item() < 0.5:
                hr = TF.hflip(hr)
            if torch.rand(1).item() < 0.5:
                hr = TF.vflip(hr)
        else:
            h, w = hr.shape[-2:]
            s = self.cfg.scale
            hr = hr[:, : h - (h % s), : w - (w % s)]

        lr = self._to_lr(hr)
        return {"lr": lr, "hr": hr, "name": hr_path.stem}
