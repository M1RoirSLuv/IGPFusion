from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    exts: Tuple[str, ...] = ("png", "jpg", "jpeg", "bmp")
    scale: int = 4
    hr_size: int = 256


class IRSRDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.hr_paths = self._scan_hr_paths()
        if len(self.hr_paths) == 0:
            raise FileNotFoundError(f"No HR images found in {cfg.root}/{cfg.split}/{cfg.hr_subdir}")

    def _scan_hr_paths(self) -> List[Path]:
        hr_dir = Path(self.cfg.root) / self.cfg.split / self.cfg.hr_subdir
        paths: List[Path] = []
        for ext in self.cfg.exts:
            paths.extend(sorted(hr_dir.glob(f"*.{ext}")))
            paths.extend(sorted(hr_dir.glob(f"*.{ext.upper()}")))
        return sorted(set(paths))

    def _read_gray(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        x = TF.to_tensor(img)  # [1,H,W], 0..1
        return x

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
        hr = self._read_gray(self.hr_paths[idx])
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
        return {"lr": lr, "hr": hr, "name": self.hr_paths[idx].stem}
