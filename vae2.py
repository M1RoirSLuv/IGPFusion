import os, torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import lpips
import numpy as np
import traceback

# ===== 配置 =====
ckpt_path = "./model/v1-5-pruned.ckpt"
data_dir = "./data/raw_512"
out_dir = "sd15_ir_vae_512_10k_lpips"
os.makedirs(out_dir, exist_ok=True)

image_size = 512  # 512x512
batch_size = 1    # 512x512图像较大，减小批次大小
lr = 5e-6
epochs = 5       # 增加训练轮数以适应10k数据
device = "cuda"

print(f"数据集: {data_dir}")
print(f"图像尺寸: {image_size}x{image_size}")
print(f"批次大小: {batch_size}")
print(f"学习率: {lr}")
print(f"训练轮数: {epochs}")

# 添加安全全局变量以支持numpy类型
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
torch.serialization.add_safe_globals([type(np.int64(0))])
torch.serialization.add_safe_globals([type(np.float64(0))])

# ===== 加载模型 =====
try:
    print("尝试从单个文件加载VAE...")
    vae = AutoencoderKL.from_single_file(
        ckpt_path, 
        torch_dtype=torch.float32,
        local_files_only=True  # 仅使用本地文件
    ).to(device)
    print("成功从单个文件加载VAE")
except Exception as e:
    print(f"直接加载失败: {e}")
    print("尝试使用替代加载方式...")
    
    # 如果直接加载失败，使用torch.load手动加载
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # 如果checkpoint是字典格式，提取state_dict
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 创建VAE实例并加载权重
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
            block_out_channels=(128, 256, 512, 512),
            latent_channels=4,
            sample_size=512,
        )
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(device)
        print("成功手动加载VAE")
    except Exception as e2:
        print(f"手动加载也失败: {e2}")
        # 创建一个新的VAE（如果没有原始模型）
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
            block_out_channels=(128, 256, 512, 512),
            latent_channels=4,
            sample_size=image_size,
        ).to(device)
        print("使用新VAE进行训练")

# 设置VAE为可训练状态
vae.requires_grad_(True)
vae.train()

# ===== 3. 初始化 LPIPS 模型 =====
try:
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    print("成功初始化LPIPS模型")
except Exception as e:
    print(f"LPIPS初始化失败: {e}")
    print("使用MSE损失替代LPIPS")
    loss_fn_vgg = None

# ===== 4. 数据准备（优化版） =====
class RobustVAEDataset(Dataset):
    def __init__(self, data_dir, size):
        self.image_paths = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        
        for f in os.listdir(data_dir):
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(data_dir, f)
                if os.path.getsize(full_path) > 1024:  # 跳过小于1KB的文件
                    self.image_paths.append(full_path)
        
        print(f"找到 {len(self.image_paths)} 张有效图像")
        
        # 数据增强策略
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # LPIPS 默认预期 [-1, 1] 范围
        ])
    
    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        attempts = 0
        while attempts < 3:
            try:
                image = Image.open(self.image_paths[idx]).convert("RGB")
                
                # 检查图像是否有效
                if image.size[0] < 32 or image.size[1] < 32:
                    print(f"图像太小，跳过: {self.image_paths[idx]}")
                    idx = (idx + 1) % len(self.image_paths)  # 尝试下一个图像
                    attempts += 1
                    continue
                    
                return self.transform(image)
            except Exception as e:
                print(f"加载图片失败 ({attempts+1}/3): {self.image_paths[idx]}, 错误: {e}")
                idx = (idx + 1) % len(self.image_paths)  # 尝试下一个图像
                attempts += 1
        
        # 如果多次尝试都失败，返回一个默认图像
        print("返回默认图像作为占位符")
        return torch.zeros(3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1])

# 创建数据集和数据加载器
try:
    dataset = RobustVAEDataset(data_dir, image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    print(f"数据加载器创建成功，共 {len(dataset)} 张图像")
except Exception as e:
    print(f"数据加载器创建失败: {e}")
    exit(1)

# ===== 5. 优化器和调度器 =====
optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
mse_criterion = torch.nn.MSELoss()

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ===== 6. 训练循环（优化版） =====
print(f"开始 LPIPS 增强训练... 共 {len(dataset)} 张图片")

best_loss = float("inf")
best_mse = float("inf")

for epoch in range(epochs):
    epoch_total_loss = 0.0
    epoch_mse_loss = 0.0
    epoch_lpips_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 检查批次数据是否有效
        if batch.nelement() == 0:
            print("跳过空批次")
            continue
            
        img = batch.to(device)
        
        # 检查图像维度是否正确
        if img.dim() != 4 or img.shape[1] != 3:
            print(f"跳过无效批次: 形状 {img.shape}")
            continue
        
        # 检查图像值范围
        if torch.isnan(img).any() or torch.isinf(img).any():
            print("跳过包含NaN或Inf的批次")
            continue
        
        try:
            # 重建流程
            latents = vae.encode(img).latent_dist.sample()
            reconstruction = vae.decode(latents).sample
            
            # 检查重构结果是否有效
            if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
                print("重构结果包含NaN或Inf，跳过此批次")
                continue
            
            # --- 组合损失函数 ---
            # 1. MSE Loss (像素级准确性)
            mse_loss = mse_criterion(reconstruction, img)
            
            # 2. LPIPS Loss (感知级清晰度)
            if loss_fn_vgg is not None:
                # LPIPS期望输入在[-1, 1]范围内，但我们的数据已经被normalize到[-1, 1]
                # 所以直接计算即可
                lpips_loss = loss_fn_vgg(reconstruction, img).mean()
                
                # 总损失：1.0 * MSE + 0.1 * LPIPS (这是一个经典的平衡比例)
                total_loss = mse_loss + 0.1 * lpips_loss
            else:
                # 如果LPIPS不可用，只使用MSE
                lpips_loss = torch.tensor(0.0, device=device)
                total_loss = mse_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_lpips_loss += lpips_loss.item()
            
            progress_bar.set_postfix({
                "mse": f"{mse_loss.item():.4f}", 
                "lpips": f"{lpips_loss.item():.4f}",
                "total": f"{total_loss.item():.4f}"
            })
            
        except Exception as e:
            print(f"批次处理出错: {e}")
            traceback.print_exc()  # 打印详细错误信息
            continue

    # 计算平均损失
    avg_total_loss = epoch_total_loss / len(dataloader)
    avg_mse_loss = epoch_mse_loss / len(dataloader)
    avg_lpips_loss = epoch_lpips_loss / len(dataloader)
    
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  平均总损失: {avg_total_loss:.6f}")
    print(f"  平均MSE损失: {avg_mse_loss:.6f}")
    print(f"  平均LPIPS损失: {avg_lpips_loss:.6f}")
    
    # 保存最佳模型（基于MSE损失）
    if avg_mse_loss < best_mse:
        best_mse = avg_mse_loss
        best_loss = avg_total_loss
        # 保存模型
        save_path = os.path.join(out_dir, "vae_best_lpips")
        os.makedirs(save_path, exist_ok=True)
        vae.save_pretrained(save_path)
        print(f"保存最佳模型，MSE: {best_mse:.6f}")
    
    # 每5个epoch保存一次检查点
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(out_dir, f"vae_checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        vae.save_pretrained(checkpoint_path)
        print(f"保存检查点: epoch {epoch+1}")
    
    # 更新学习率
    scheduler.step()

# 保存最终模型
final_save_path = os.path.join(out_dir, "vae_final_lpips")
os.makedirs(final_save_path, exist_ok=True)
vae.save_pretrained(final_save_path)

print(f"VAE训练完成！")
print(f"最佳模型保存在: {os.path.join(out_dir, 'vae_best_lpips')}")
print(f"最终模型保存在: {final_save_path}")
