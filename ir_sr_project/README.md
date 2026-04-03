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
python ir_sr_project/infer.py --ckpt outputs/ir_sr/best.pth --input path/to/lr.png --output out.png
```

## Notes
- `models/teacher_adapter.py` is only used in training when `train.use_teacher_distill=true`.
- `infer.py` never imports teacher/CLIP modules.
- If LPIPS package is unavailable, code falls back to L1 perceptual proxy with a warning.

## Optional Quality Optimization Paths

You can further improve quality with either of the following training-only strategies:

1. Prompt-guided optimization:
   - Encode SR image with CLIP image encoder.
   - Compute similarity against positive/negative infrared prompts.
   - Add a prompt ranking loss to bias outputs toward high-quality thermal semantics.

2. DifIISR-style gradient guiding (lightweight variant):
   - Keep student forward structure unchanged.
   - Add a guidance loss gradient term from frequency/perceptual priors at intermediate stages.
   - Use low guidance weight and late-epoch scheduling to avoid over-constraining texture.

Inference remains unchanged (student-only), so deployment cost is still low.


## Enable CLIP Prompt Branch in Training

Set in config:

```yaml
train:
  use_prompt_loss: true
loss:
  w_prompt: 0.02
prompt:
  model_name_or_path: model/clip-vit-large-patch14
  positive_prompt: "a high quality infrared image with clear thermal edges"
  negative_prompt: "a blurry noisy low quality infrared image"
  margin: 0.1
```

The training loop will compute CLIP image/text cosine scores and add a ranking loss:
`L_prompt = relu(margin - s_pos + s_neg)`

This branch is training-only; inference remains student-only.
