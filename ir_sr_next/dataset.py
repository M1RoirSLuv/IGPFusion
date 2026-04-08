from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


@dataclass
class DatasetConfig:
    root: str
    split: str
    hr_subdir: str = "HR"
    lr_subdir: str = "LR"
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    use_precomputed_lr: bool = True
    scale: int = 4
    hr_size: int = 256


class InfraredSRDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        root = Path(cfg.root)
        split = cfg.split

        self.hr_dir = root / split / cfg.hr_subdir
        self.lr_dir = root / split / cfg.lr_subdir

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"Missing HR directory: {self.hr_dir}")

        self.hr_files: List[Path] = sorted(
            [p for p in self.hr_dir.iterdir() if p.suffix.lower() in cfg.exts]
        )
        if not self.hr_files:
            raise RuntimeError(f"No HR files found in {self.hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_files)

    @staticmethod
    def _load_gray(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        return TF.to_tensor(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr_path = self.hr_files[idx]
        hr = self._load_gray(hr_path)

        if self.cfg.use_precomputed_lr:
            lr_path = self.lr_dir / hr_path.name
            if not lr_path.exists():
                raise FileNotFoundError(f"Missing LR file for {hr_path.name}: {lr_path}")
            lr = self._load_gray(lr_path)
        else:
            h, w = hr.shape[-2:]
            lr = TF.resize(
                hr,
                [max(1, h // self.cfg.scale), max(1, w // self.cfg.scale)],
                antialias=True,
            )

        if self.cfg.hr_size > 0:
            hr = TF.center_crop(hr, [self.cfg.hr_size, self.cfg.hr_size])
            lr_size = self.cfg.hr_size // self.cfg.scale
            lr = TF.center_crop(lr, [lr_size, lr_size])

        return {"lr": lr.clamp(0, 1), "hr": hr.clamp(0, 1), "name": hr_path.name}
