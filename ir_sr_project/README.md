# IR-SR Distill

A standalone infrared super-resolution/enhancement project split from the fusion repo.

## Goals
- Keep inference independent (student network only).
- Use LPIPS as perceptual loss.
- Optionally borrow DifIISR-style frequency prior (`L_freq`) without introducing full diffusion sampling complexity.
- Optionally distill from your SD1.5+LoRA infrared generative teacher during training.

## Project Structure

```text
ir_sr_project/
  configs/base.yaml
  datasets/ir_sr_dataset.py
  models/corple_student.py
  models/teacher_adapter.py
  models/losses.py
  train.py
  infer.py
  scripts/train.sh
```

## Data Layout

```text
data_root/
  train/
    HR/*.png
  val/
    HR/*.png
```

LR is generated online by bicubic downsampling.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install lpips pyyaml pillow tqdm
python ir_sr_project/train.py --config ir_sr_project/configs/base.yaml
python ir_sr_project/infer.py --ckpt outputs/ir_sr_mvp/best.pth --input path/to/lr.png --output out.png
```

## Notes
- `models/teacher_adapter.py` is only used in training when `train.use_teacher_distill=true`.
- `infer.py` never imports teacher/CLIP modules.
- If LPIPS package is unavailable, code falls back to L1 perceptual proxy with a warning.
