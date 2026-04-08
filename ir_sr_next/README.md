# IR SR Project V2 (DifIISR-aligned + CLIP prompt adapter)

这版是面向你当前需求的折中实现：

1. **保留 DifIISR 主线**：IR diffusion prior + gradient guidance + 多损失约束。
2. **加入 CoRPLE 可借鉴点**：引入 `CLIP + PromptTokenAdapter`，可学习 token 拼接到文本条件上。
3. **红外生成式模型强绑定**：`vae_path`、`diffusion_model_path` 必填，不做 fallback。

## 和论文“1:1 DifIISR”的关系

- 当前实现是 **DifIISR-aligned**（思想对齐版），不是完全复刻其整套扩散采样/反演循环。
- 如果你已经有 DifIISR 原始代码，建议直接以其采样主干为准，把本仓的 IR VAE、数据管线和 CLIP token adapter 合入。

## CoRPLE 可借鉴点（已落地）

- Prompt learning 主机制（token adapter）已实现：
  - `PromptTokenAdapter`：可学习 prompt token
  - `PromptEncoder`：CLIP 文本编码 + token adapter 拼接
  - prior 提取时作为 UNet 的 `encoder_hidden_states`

## 训练

```bash
python -m ir_sr_next.train --config ir_sr_next/configs/difiisr_ir_prior_x4.yaml
```

## 推理

```bash
python -m ir_sr_next.infer \
  --ckpt outputs/ir_sr_next_x4/best.pth \
  --input path/to/lr.png \
  --output path/to/sr.png
```

## 结构

- `ir_sr_next/models/diffusion_prior_sr.py`: 主模型（diffusion prior + gradient guidance + CLIP token adapter）
- `ir_sr_next/dataset.py`: 数据集
- `ir_sr_next/train.py`: 训练（pix + grad + prior + freq + LPIPS）
- `ir_sr_next/infer.py`: 推理
- `ir_sr_next/configs/difiisr_ir_prior_x4.yaml`: 默认配置
