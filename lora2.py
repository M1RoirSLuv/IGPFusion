import os, torch, torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import traceback

# 路径配置
ckpt_path = "./model/v1-5-pruned.ckpt"
clip_path = "./model/clip-vit-large-patch14"
data_dir = "./data/raw_512"
out_dir = "sd15_ir_lora_512_10k_fixed"
os.makedirs(out_dir, exist_ok=True)

# 参数设置
image_size = 512
batch_size = 1
lr = 1e-4
epochs = 10
warmup_epochs = 3
device = "cuda"

print(f"=== 512x512红外LoRA微调 ===")
print(f"数据集: {data_dir}, 分辨率: {image_size}x{image_size}")

# ===== 加载基础模型 =====
print("正在加载基础模型...")
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # 构建UNet
    unet = UNet2DConditionModel(
        sample_size=image_size // 8,
        in_channels=4, out_channels=4,
        layers_per_block=2,
        block_out_channels=[320, 640, 1280, 1280],
        down_block_types=["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
        up_block_types=["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
        cross_attention_dim=768,
        attention_head_dim=8,
    )
    unet_dict = unet.state_dict()
    pretrained_dict = {k.replace('model.diffusion_model.', ''): v for k, v in state_dict.items() 
                       if k.startswith('model.diffusion_model') and k.replace('model.diffusion_model.', '') in unet_dict}
    unet_dict.update(pretrained_dict)
    unet.load_state_dict(unet_dict)

    # 构建VAE
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=(128, 256, 512, 512),
        latent_channels=4,
        sample_size=image_size,
    )
    vae_dict = vae.state_dict()
    pretrained_vae_dict = {k.replace('first_stage_model.', ''): v for k, v in state_dict.items() 
                           if k.startswith('first_stage_model') and k.replace('first_stage_model.', '') in vae_dict}
    vae_dict.update(pretrained_vae_dict)
    vae.load_state_dict(vae_dict)

    # 加载CLIP
    tokenizer = CLIPTokenizer.from_pretrained(clip_path, local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(clip_path, local_files_only=True)

    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDPMScheduler.from_config({"num_train_timesteps": 1000}),
        safety_checker=None, feature_extractor=None,
    )
    print("基础模型构建完成")

except Exception as e:
    print(f"基础模型加载失败: {e}")
    exit(1)

# ===== 加载训练好的VAE =====
trained_vae_path = "./sd15_ir_vae_512_10k_lpips/vae_best_lpips"
if os.path.exists(trained_vae_path):
    try:
        trained_vae = AutoencoderKL.from_pretrained(trained_vae_path, torch_dtype=torch.float16)
        pipe.vae = trained_vae
        print(f"已加载训练VAE: {trained_vae_path}")
    except Exception as e:
        print(f"加载训练VAE失败: {e}, 使用基础VAE")
else:
    print(f"未找到训练VAE: {trained_vae_path}")

# 移动到设备并冻结
pipe = pipe.to(device, dtype=torch.float16)
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)  # 先冻结，LoRA注入后再开启部分

# ===== 3. 数据集 =====
class InfraredDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"找到 {len(self.image_paths)} 张图像")
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if img.size != (self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]):
                img = img.resize((self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))
            return {"pixel_values": self.transform(img)}
        except Exception as e:
            print(f"加载失败: {e}")
            return {"pixel_values": torch.ones(3, image_size, image_size) * 0.5}

dataset = InfraredDataset(data_dir, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

# ===== LoRA层定义  =====
class SafeLoRA(nn.Module):
    def __init__(self, layer, rank=8):
        super().__init__()
        self.layer = layer
        self.rank = rank
        
        if isinstance(layer, nn.Linear):
            self.down = nn.Linear(layer.in_features, rank, bias=False)
            self.up = nn.Linear(rank, layer.out_features, bias=False)
        elif isinstance(layer, nn.Conv2d):
            self.down = nn.Conv2d(layer.in_channels, rank, 1, bias=False)
            self.up = nn.Conv2d(rank, layer.out_channels, 1, bias=False)
        else:
            raise ValueError(f"Unsupported layer: {type(layer)}")
        
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        
        # 立即移动到与原层相同的设备
        self.to(layer.weight.device)
        
    def forward(self, x):
        # 保持输入输出dtype一致，内部计算用float32
        input_dtype = x.dtype
        x = x.to(torch.float32)
        residual = self.layer(x)
        lora_out = self.up(self.down(x))
        return (residual + lora_out).to(input_dtype)

# ===== 注入LoRA =====
print("正在注入LoRA层...")
lora_params = []
injected_count = 0

for name, module in pipe.unet.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        if any(x in name for x in ['downsample', 'upsample', 'time_emb_proj']):
            continue
        if hasattr(module, 'layer'): # 防止重复
            continue
            
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = pipe.unet
        for part in parent_name.split('.'):
            if part: parent = getattr(parent, part)
            
        lora_wrapper = SafeLoRA(module, rank=8) # Rank设为8以节省显存
        setattr(parent, child_name, lora_wrapper)
        
        lora_params.extend(list(lora_wrapper.down.parameters()))
        lora_params.extend(list(lora_wrapper.up.parameters()))
        injected_count += 1

print(f"成功注入 {injected_count} 个LoRA模块，共 {len(lora_params)} 个参数")

# 再次确认所有LoRA参数都在GPU上
for p in lora_params:
    if not p.is_cuda:
        p.data = p.data.to(device)

# 开启LoRA参数的梯度
for p in lora_params:
    p.requires_grad = True

# ===== 优化器与调度器 =====
optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs * len(dataloader))
main_sched = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs) * len(dataloader))
scheduler = SequentialLR(optimizer, [warmup_sched, main_sched], [warmup_epochs * len(dataloader)])

scaler = torch.cuda.amp.GradScaler()

# 预计算空提示
with torch.no_grad():
    empty_prompt_embeds = pipe.text_encoder(
        pipe.tokenizer([""], return_tensors="pt", padding=True).input_ids.to(device)
    )[0].to(dtype=torch.float16)

# ===== 训练循环 (修复AMP逻辑) =====
print("开始训练...")
pipe.unet.train()
best_loss = float('inf')

def save_lora(path):
    state = {}
    for name, param in pipe.unet.named_parameters():
        if 'down.' in name or 'up.' in name:
            state[name] = param
    torch.save(state, path)

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for step, batch in enumerate(progress_bar):
        try:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # --- 核心修复：AMP上下文管理 ---
            with torch.amp.autocast('cuda', dtype=torch.float16):
                noise_pred = pipe.unet(noisy_latents, timesteps, empty_prompt_embeds.expand(latents.shape[0], -1, -1)).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪和更新流程
            # 1. Unscale gradients (只调用一次)
            scaler.unscale_(optimizer)
            
            # 2. Clip gradients
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            
            # 3. Step optimizer
            scaler.step(optimizer)
            
            # 4. Update scaler
            scaler.update()
            
            # 5. Update scheduler
            scheduler.step()
            
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
            
        except Exception as e:
            print(f"\n批次错误: {e}")
            # 如果发生严重错误，重置scaler以防状态损坏
            scaler.update() 
            continue

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(out_dir, "best_lora.pt")
        save_lora(save_path)
        print(f"保存最佳模型: {save_path}")
    
    if (epoch + 1) % 5 == 0:
        save_path = os.path.join(out_dir, f"lora_epoch_{epoch+1}.pt")
        save_lora(save_path)

# 保存最终模型
save_lora(os.path.join(out_dir, "final_lora.pt"))
print("训练完成！")
