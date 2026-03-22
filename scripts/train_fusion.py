import argparse

from fusion.configs import TrainConfig
from fusion.trainer import train


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Modular dual-VAE fusion training")
    p.add_argument("--sd_ckpt_path", type=str, default="./model/v1-5-pruned.ckpt")
    p.add_argument("--ir_vae_path", type=str, default="./sd15_ir_vae_512_10k_lpips/vae_best_lpips")
    p.add_argument("--ir_data_dir", type=str, default="/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Ir")
    p.add_argument("--vis_data_dir", type=str, default="/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Vis")
    p.add_argument("--output_dir", type=str, default="./dual_vae_fusion_runs")
    p.add_argument("--vis_vae_dir", type=str, default=None)
    p.add_argument("--vis_check_image", type=str, default=None)
    p.add_argument("--vae_threshold", type=int, default=10)
    p.add_argument("--start_stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--no_force_resize", action="store_true")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_stage1", type=float, default=1e-4)
    p.add_argument("--lr_stage2", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs_stage1", type=int, default=10)
    p.add_argument("--epochs_stage2", type=int, default=30)
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--feature_channels", type=int, default=64)
    p.add_argument("--recon_weight_ir", type=float, default=1.0)
    p.add_argument("--recon_weight_vis", type=float, default=1.0)
    p.add_argument("--decomp_weight_stage1", type=float, default=0.5)
    p.add_argument("--intensity_weight", type=float, default=1.0)
    p.add_argument("--grad_weight", type=float, default=0.1)
    p.add_argument("--stage2_vis_recon_ratio", type=float, default=0.6)
    p.add_argument("--stage2_ir_gray_ratio", type=float, default=0.35)
    p.add_argument("--stage2_ir_salient_ratio", type=float, default=0.3)
    p.add_argument("--ir_saliency_tau", type=float, default=0.7)
    p.add_argument("--ir_saliency_temp", type=float, default=0.08)
    p.add_argument("--fusion_max_int_weight", type=float, default=1.0)
    p.add_argument("--fusion_max_grad_weight", type=float, default=3.0)
    p.add_argument("--fusion_color_sal_weight", type=float, default=0.0)
    p.add_argument("--fusion_color_bg_weight", type=float, default=0.0)
    p.add_argument("--stage2_ssim_weight", type=float, default=3.0)
    p.add_argument("--saliency_weight", type=float, default=0.08)
    p.add_argument("--saliency_ir_target_ratio", type=float, default=0.7)
    p.add_argument("--stage2_ir_edge_weight", type=float, default=2.0)
    p.add_argument("--stage2_vis_edge_bg_weight", type=float, default=1.5)
    p.add_argument("--stage2_vis_hf_weight", type=float, default=1.0)
    p.add_argument("--stage2_vis_hf_bg_weight", type=float, default=1.5)
    p.add_argument("--stage2_vis_hf_sal_weight", type=float, default=0.35)
    p.add_argument("--stage2_vis_lap_weight", type=float, default=0.6)
    p.add_argument("--stage2_vis_lap_bg_weight", type=float, default=1.0)
    p.add_argument("--stage2_vis_lap_sal_weight", type=float, default=0.15)
    p.add_argument("--stage2_edge_focus_weight", type=float, default=1.2)
    p.add_argument("--stage2_vis_bg_int_weight", type=float, default=1.5)
    p.add_argument("--stage2_salient_color_suppress", type=float, default=0.5)
    p.add_argument("--stage2_unfreeze_ir_branches", type=str2bool, default=True)
    p.add_argument("--stage2_unfreeze_vis_align", type=str2bool, default=False)
    # p.add_argument("--haze_loss_weight", type=float, default=0.0)
    # p.add_argument("--haze_brightness_tau", type=float, default=0.55)
    # p.add_argument("--haze_grad_tau", type=float, default=0.08)
    # p.add_argument("--haze_mask_temp", type=float, default=0.10)
    p.add_argument("--decomp_weight_stage2", type=float, default=0.2)
    p.add_argument("--epsilon_decomp", type=float, default=1.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--output_gray_only", action="store_true", default=True)
    p.add_argument("--keep_vis_colorize", action="store_true")
    a = p.parse_args()
    cfg_dict = vars(a).copy()
    cfg_dict["amp"] = not cfg_dict.pop("no_amp")
    cfg_dict["force_resize"] = not cfg_dict.pop("no_force_resize")
    keep_vis_colorize = cfg_dict.pop("keep_vis_colorize")
    cfg_dict["output_gray_only"] = False if keep_vis_colorize else cfg_dict["output_gray_only"]
    return TrainConfig(**cfg_dict)


if __name__ == "__main__":
    train(parse_args())
