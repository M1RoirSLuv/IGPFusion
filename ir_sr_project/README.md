# IR-SR Distill

A standalone infrared super-resolution/enhancement project split from the fusion repo.


## PR conflict note

If GitHub web editor reports too many conflicts, do not edit in web UI.
Use local git merge/rebase/cherry-pick and push a refreshed PR branch instead.

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
  tools/generate_synthetic_hr.py
  tools/generate_paired_from_generative.py
  prompts/ir_realistic_prompts.txt
  prompts/ir_negative_prompts.txt
```

## Data Layout

```text
data_root/
  train/
    HR/*.png
    LR/*.png        # optional, when use_precomputed_lr=true
  val/
    HR/*.png
    LR/*.png        # optional, when use_precomputed_lr=true
```



If `use_precomputed_lr=false` (default), LR is generated online by bicubic downsampling from HR.
If `use_precomputed_lr=true`, dataset will load paired LR/HR from disk and match files by the same filename stem.






## Ready-to-use infrared prompt set

A curated prompt set is provided to reduce unrealistic thermal modality artifacts:

- `ir_sr_project/prompts/ir_realistic_prompts.txt` (one scene prompt per line)
- `ir_sr_project/prompts/ir_negative_prompts.txt` (style/artifact negatives)

Use with generation tools, for example:

```bash
python ir_sr_project/tools/generate_paired_from_generative.py \
  --base_model /path/to/generative \
  --prompt_file ir_sr_project/prompts/ir_realistic_prompts.txt \
  --negative_prompt "$(head -n 1 ir_sr_project/prompts/ir_negative_prompts.txt)" \
  --out_root data/dataset/train \
  --hr_size 512 --scale 4 --to_grayscale
```



## Stage1 -> Stage2 training (synthetic then real FLIR-IISR)

If you train stage1 on generated data and then stage2 on real FLIR-IISR, use:

```yaml
train:
  stage_name: stage2_flir
  resume_ckpt: outputs/ir_sr/best_stage1_synth.pth
  resume_mode: finetune
  reset_best_on_resume: true
```

Checkpoint behavior:
- `stage_name` empty: saves `best.pth` / `last.pth`.
- `stage_name` set: saves `best_<stage_name>.pth` / `last_<stage_name>.pth` (e.g. `best_stage2_flir.pth`).

This ensures stage2 best selection is based on stage2 validation only.


A ready config is provided: `ir_sr_project/configs/stage2_from_stage1.yaml`

Run:

```bash
python ir_sr_project/train.py --config ir_sr_project/configs/stage2_from_stage1.yaml
```

## Using a merged SD1.5-IR model (no separate `lora_path`)

If your LoRA has already been merged and your folder looks like:
`model_index.json + unet/ + vae/ + text_encoder/ + tokenizer/ + scheduler/`,
just treat that folder as the base model path.

- Data generation tools: pass `--base_model /path/to/generative` and omit `--lora_path`.
- Teacher distillation: set `train.use_teacher_distill=true` and `train.teacher_path: /path/to/generative`.

This matches your current directory structure in the screenshot.

## CoRPLE / DifIISR Reuse Status

- Current code is **not** a line-by-line direct reuse of CoRPLE or DifIISR repos.
- You can directly reuse their code only if you follow each repository's license requirements and keep attribution notices.
- The student network is a **CoRPLE-inspired lightweight design**.
- The frequency-domain loss is a **DifIISR-inspired idea** (lightweight adaptation), not their full diffusion reverse-process guidance.

## Do You Need Generative Model Outputs as HR?

Short answer: **No, not required**.

- For super-resolution training, the standard and preferred setup is: use real infrared HR images as ground truth, then bicubic downsample online to make LR inputs.
- Your generative model is optional, mainly for:
  1) data augmentation (synthetic HR),
  2) teacher feature distillation (training-time prior).

Recommended priority:
1. Real HR dataset first (main training data).
2. Synthetic HR as auxiliary only (mixed in with lower sampling ratio).



## Use Generated LR from Your Generative Model

Yes, you can train with your generated LR, but supervised SR still needs HR targets.

Recommended paired layout:

```text
data_root/
  train/
    HR/img_0001.png
    LR/img_0001.png
  val/
    HR/img_1001.png
    LR/img_1001.png
```

Enable paired LR loading in config:

```yaml
data:
  use_precomputed_lr: true
  hr_subdir: HR
  lr_subdir: LR
```

Important:
- LR-only data (without HR) is not enough for this supervised training loop.
- If generated LR has no matched HR, use it only for unsupervised/self-training extensions (not included in current code).



### Fastest way: generate paired LR/HR in one command

If your model is SD1.5 infrared LoRA, run:

```bash
python ir_sr_project/tools/generate_paired_from_generative.py \
  --base_model path_or_hf_id_to_sd15 \
  --lora_path path/to/your_ir_lora \
  --prompt_file data/prompts_ir.txt \
  --out_root data/dataset/train \
  --hr_size 512 \
  --scale 4 \
  --num_images_per_prompt 6 \
  --to_grayscale
```

This writes:

```text
data/dataset/train/HR/img_000000.png
data/dataset/train/LR/img_000000.png
```

Then in config set:

```yaml
data:
  use_precomputed_lr: true
  hr_subdir: HR
  lr_subdir: LR
```

Tips:
- Use real infrared data as primary supervision; generated pairs are auxiliary.
- Tune `--blur_sigma` and `--noise_std` to mimic your sensor degradation.

## Optional: Generate Synthetic HR with Your SD1.5+LoRA

You can create extra synthetic HR images with:

```bash
python ir_sr_project/tools/generate_synthetic_hr.py \
  --base_model path_or_hf_id_to_sd15 \
  --lora_path path/to/your_lora \
  --prompt_file data/prompts_ir.txt \
  --output_dir data/synthetic_hr \
  --num_images_per_prompt 4
```

`prompt_file` format (one prompt per line):

```text
a high-quality infrared surveillance scene, clear thermal edges
a thermal pedestrian scene at night, sharp structure, low noise
```

Then place generated images under your HR folder (or a parallel synthetic split) and keep real data dominant.

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
