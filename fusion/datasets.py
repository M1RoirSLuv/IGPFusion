from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedInfraredVisibleDataset(Dataset):
    def __init__(self, ir_dir: str, vis_dir: str, image_size: int, force_resize: bool = True, logger=None):
        self.ir_dir = Path(ir_dir)
        self.vis_dir = Path(vis_dir)
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        ir_files = {p.stem: p for p in self.ir_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext}
        vis_files = {p.stem: p for p in self.vis_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext}

        common_stems = sorted(set(ir_files) & set(vis_files))
        if not common_stems:
            raise RuntimeError(f"No paired files found. IR dir: {ir_dir}, VIS dir: {vis_dir}")

        self.pairs: List[Tuple[Path, Path, str]] = [(ir_files[s], vis_files[s], s) for s in common_stems]
        self.force_resize = bool(force_resize)
        self.image_size = int(image_size)

        if self.force_resize:
            self.transform_vis = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.transform_ir = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.transform_vis = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.transform_ir = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        if logger:
            logger.info("Loaded %d paired samples. force_resize=%s image_size=%d", len(self.pairs), self.force_resize, self.image_size)

    @staticmethod
    def _center_crop_to_multiple_of_8(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = w - (w % 8)
        new_h = h - (h % 8)
        if new_w < 8 or new_h < 8:
            raise RuntimeError(f"Image too small for VAE after crop: {img.size}")
        if (new_w, new_h) == (w, h):
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def _align_pair_without_resize(self, ir_img: Image.Image, vis_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if ir_img.size != vis_img.size:
            vis_img = vis_img.resize(ir_img.size, resample=Image.BICUBIC)
        ir_img = self._center_crop_to_multiple_of_8(ir_img)
        vis_img = self._center_crop_to_multiple_of_8(vis_img)
        return ir_img, vis_img

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ir_path, vis_path, stem = self.pairs[idx]
        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("RGB")
        if not self.force_resize:
            ir_img, vis_img = self._align_pair_without_resize(ir_img, vis_img)
        ir = self.transform_ir(ir_img).repeat(3, 1, 1)
        vis = self.transform_vis(vis_img)
        return {"ir": ir, "vis": vis, "stem": stem}
