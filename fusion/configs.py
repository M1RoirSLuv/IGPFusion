from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    sd_ckpt_path: str
    ir_vae_path: str
    ir_data_dir: str
    vis_data_dir: str
    output_dir: str

    vis_vae_dir: Optional[str] = None
    vis_check_image: Optional[str] = None
    vae_threshold: int = 10
    start_stage: int = 1
    resume_ckpt: Optional[str] = None

    image_size: int = 512
    force_resize: bool = True
    batch_size: int = 2
    num_workers: int = 4

    lr_stage1: float = 1e-4
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4

    epochs_stage1: int = 10
    epochs_stage2: int = 30

    latent_channels: int = 4
    feature_channels: int = 64

    recon_weight_ir: float = 1.0
    recon_weight_vis: float = 1.0
    decomp_weight_stage1: float = 0.5

    intensity_weight: float = 1.0
    decomp_weight_stage2: float = 0.2
    stage2_vis_recon_ratio: float = 0.6
    stage2_ir_gray_ratio: float = 0.35
    stage2_ir_salient_ratio: float = 0.3
    ir_saliency_tau: float = 0.7
    ir_saliency_temp: float = 0.08
    fusion_max_int_weight: float = 1.0
    fusion_max_grad_weight: float = 3.0
    stage2_ssim_weight: float = 3.0
    stage2_vis_hf_weight: float = 1.0
    stage2_vis_lap_weight: float = 0.6
    stage2_edge_focus_weight: float = 1.2
    stage2_unfreeze_ir_branches: bool = True
    stage2_unfreeze_vis_align: bool = False
    # haze_loss_weight: float = 0.0
    # haze_brightness_tau: float = 0.55
    # haze_grad_tau: float = 0.08
    # haze_mask_temp: float = 0.10
    epsilon_decomp: float = 1.01
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42
    save_every: int = 1
    output_gray_only: bool = True

    # IR prior (frozen UNet fine-tuned on IR images)
    ir_prior_unet_path: Optional[str] = None
    ir_prior_weight: float = 0.1
    ir_prior_layers: str = "1,2"
