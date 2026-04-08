# IR SR Project V2 (DifIISR-aligned + CLIP prompt adapter)

在你要求下，当前版本保留并聚焦：

- **完整扩散子模块拆分（schedule / respace / gaussian_diffusion / inversion / sampler）**
- **扩散反演（DDIMInversion）**
- **逐 timestep 采样循环（scheduler + respace）**
- **每个采样步注入 guidance 梯度**
- **SR 主训练（使用你已有红外先验）**

## 训练（仅 SR）

```bash
python -m ir_sr_next.train --config ir_sr_next/configs/difiisr_ir_prior_x4.yaml
```

## 推理

### 1) SR 快速推理

```bash
python -m ir_sr_next.infer \
  --ckpt outputs/ir_sr_next_x4/best.pth \
  --input path/to/lr.png \
  --output path/to/sr.png \
  --mode sr
```

### 2) DDIM 反演+采样推理

```bash
python -m ir_sr_next.infer \
  --ckpt outputs/ir_sr_next_x4/best.pth \
  --input path/to/lr.png \
  --output path/to/sr.png \
  --mode ddim \
  --prompt "a high quality infrared image with clear thermal edges"
```

## 关键结构

- `ir_sr_next/diffusion/schedule.py`: beta schedule
- `ir_sr_next/diffusion/respace.py`: timestep respacing
- `ir_sr_next/diffusion/gaussian_diffusion.py`: diffusion 核心方程
- `ir_sr_next/diffusion/inversion.py`: DDIM inversion
- `ir_sr_next/diffusion/sampler.py`: step-wise guidance 注入采样器
- `ir_sr_next/models/diffusion_prior_sr.py`: SR主网络（diffusion prior + gradient guidance + CLIP token adapter）
- `ir_sr_next/train.py`: SR训练
- `ir_sr_next/infer.py`: 推理（sr/ddim 双模式）
