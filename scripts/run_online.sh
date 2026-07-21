#!/usr/bin/env bash
# Set these to your local paths (or export before running):
: "${PROJ_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${DATA_ROOT:=${PROJ_ROOT}/data}"
: "${OUTPUTS_ROOT:=${PROJ_ROOT}/outputs}"

set -euo pipefail

# === 项目根目录（包含 scripts/ 与 uda_core/）===
PROJ_ROOT="${PROJ_ROOT}"
# 安全设置 PYTHONPATH（兼容 set -u）
export PYTHONPATH="${PROJ_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python3 "${PROJ_ROOT}/train_online_end2end.py" \
  --npy_src ${DATA_ROOT}/MRNet-v1.0/knees_npy \
  --csv_src ${DATA_ROOT}/MRNet-v1.0/train_0.csv \
  --csv_src_val ${DATA_ROOT}/MRNet-v1.0/valid_0.csv \
  --npy_tgt ${DATA_ROOT}/KneeMRI/knees_npy \
  --csv_tgt ${DATA_ROOT}/KneeMRI/train_0.csv \
  --plane sagittal --resize 224 \
  --id_col_src case_id --label_col_src label --single_file_case_src --id_zero_pad_src 0 \
  --id_col_tgt case_id --single_file_case_tgt --id_zero_pad_tgt 0 \
  --backbone custom_resnet50_space --pretrained imagenet \
  --init_from ${OUTPUTS_ROOT}/retrain_recall/src_only_custom_rsa_lr1e-4_wd4e-4_1104/custom_resnet50_space_best.pth \
  --bs_src 32 --bs_tgt 32 \
  --lr 1e-4 \
  --wd 6e-4 \
  --rounds 60 \
  --epochs_contrast 6 \
  --K 1 --Kmax 1 \
  --ema_m 0.95 \
  --proto_m 0.97 \
  --tau_proto 0.07 \
  --use_nmf_pseudo \
  --nmf_assign_iters 100 --nmf_pool_size 3000 --nmf_max_iter 200 \
  --mask_cross_class_update \
  --tau_base 0.82 --tau_min 0.60 --cover_max 0.75 \
  --proto_init nmf --nmf_init_beta frobenius --nmf_init_max_iter 150 \
  --save_dir ${PROJ_ROOT}/pipe_out/1111_lr1e-4_wd6e-4_nmf_dynamic_freematch_0211
