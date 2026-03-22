#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_saliency_ablation.sh
# Optional overrides:
#   IR_DATA_DIR=... VIS_DATA_DIR=... RESUME_CKPT=... OUT_ROOT=... CONDA_ENV=fusionfix bash scripts/run_saliency_ablation.sh

WORKDIR="/public/home/xuhaoyuan/tmp/ddpm_demo/Thermal_Diffusion_Project"
CONDA_ENV="${CONDA_ENV:-fusionfix}"
IR_DATA_DIR="${IR_DATA_DIR:-/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Ir}"
VIS_DATA_DIR="${VIS_DATA_DIR:-/public/home/xuhaoyuan/tmp/ivifdataset/M3FD/Vis}"
RESUME_CKPT="${RESUME_CKPT:-./dual_vae_fusion_runs_stage2_4200_attn_detail_v2/stage2_best.pt}"
OUT_ROOT="${OUT_ROOT:-./ablation_saliency}"

cd "${WORKDIR}"
mkdir -p "${OUT_ROOT}"

run_one () {
  local tag="$1"
  local saliency_weight="$2"
  local out_dir="${OUT_ROOT}/${tag}"
  local log_file="${OUT_ROOT}/${tag}.log"

  echo "[RUN] ${tag} saliency_weight=${saliency_weight}"
  conda run -n "${CONDA_ENV}" python -u -m scripts.train_fusion \
    --sd_ckpt_path ./model/v1-5-pruned.ckpt \
    --ir_vae_path ./sd15_ir_vae_512_10k_lpips/vae_best_lpips \
    --ir_data_dir "${IR_DATA_DIR}" \
    --vis_data_dir "${VIS_DATA_DIR}" \
    --output_dir "${out_dir}" \
    --start_stage 2 \
    --resume_ckpt "${RESUME_CKPT}" \
    --epochs_stage2 12 \
    --no_force_resize \
    --batch_size 1 \
    --num_workers 3 \
    --lr_stage2 5e-6 \
    --output_gray_only \
    --fusion_max_int_weight 1.0 \
    --fusion_max_grad_weight 1.7 \
    --intensity_weight 1.0 \
    --stage2_ssim_weight 2.2 \
    --saliency_weight "${saliency_weight}" \
    --stage2_vis_bg_int_weight 2.8 \
    --stage2_vis_edge_bg_weight 3.2 \
    --stage2_vis_hf_weight 1.8 \
    --stage2_vis_hf_bg_weight 2.8 \
    --stage2_vis_lap_weight 0.9 \
    --stage2_vis_lap_bg_weight 1.4 \
    --stage2_ir_edge_weight 1.4 \
    --stage2_edge_focus_weight 0.6 \
    --decomp_weight_stage2 0.12 \
    --stage2_unfreeze_ir_branches true \
    --stage2_unfreeze_vis_align true \
    --grad_clip 1.0 \
    > "${log_file}" 2>&1

  echo "[DONE] ${tag} -> ${out_dir}"
}

# A/B/C controls
run_one "A_base" "0.0"
run_one "B_saliency_03" "0.3"
run_one "C_saliency_06" "0.6"

echo "All ablation runs completed. Logs under ${OUT_ROOT}."
