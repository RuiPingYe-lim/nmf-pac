#!/usr/bin/env bash
# Threshold ablation for the reverse direction: KneeMRI -> MRNet-v1.0.
# Set these to your local paths (or export before running):
: "${PROJ_ROOT:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${DATA_ROOT:=${PROJ_ROOT}/data}"
: "${OUTPUTS_ROOT:=${PROJ_ROOT}/outputs}"

set -euo pipefail

export PYTHONPATH="${PROJ_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_ENTRY="${PROJ_ROOT}/train_online_end2end.py"
OUT_ROOT="${PROJ_ROOT}/pipe_out/ablation_knee2mrnet"

COMMON_ARGS=(
  # KneeMRI -> MRNet-v1.0 (source=KneeMRI, target=MRNet-v1.0).
  # source validation = source's own validation split; target test = target's test split.
  --npy_src ${DATA_ROOT}/KneeMRI/knees_npy
  --csv_src ${DATA_ROOT}/KneeMRI/train_0.csv
  --csv_src_val ${DATA_ROOT}/KneeMRI/valid_0.csv
  --npy_tgt ${DATA_ROOT}/MRNet-v1.0/knees_npy
  --csv_tgt ${DATA_ROOT}/MRNet-v1.0/train_0.csv
  --npy_tgt_test ${DATA_ROOT}/MRNet-v1.0/knees_npy
  --csv_tgt_test ${DATA_ROOT}/MRNet-v1.0/test_0.csv
  --plane sagittal
  --resize 224
  --id_col_src case_id
  --label_col_src label
  --single_file_case_src
  --id_zero_pad_src 0
  --id_col_tgt case_id
  --single_file_case_tgt
  --id_zero_pad_tgt 0
  --id_col_tgt_test case_id
  --label_col_tgt_test label
  --single_file_case_tgt_test
  --id_zero_pad_tgt_test 0
  --backbone custom_resnet50_space
  --pretrained imagenet
  --init_from ${OUTPUTS_ROOT}/src_only_kneemri/custom_resnet50_space_best.pth
  --bs_src 32
  --bs_tgt 32
  --lr 2e-4
  --wd 3e-4
  --rounds 60
  --epochs_contrast 6
  --K 1
  --Kmax 1
  --ema_m 0.95
  --proto_m 0.97
  --tau_proto 0.07
  --nmf_assign_iters 100
  --nmf_pool_size 3000
  --nmf_max_iter 200
  --mask_cross_class_update
  --tau_base 0.82
  --tau_min 0.60
  --cover_max 0.75
  --proto_init nmf
  --nmf_init_beta frobenius
  --nmf_init_max_iter 150
)

run_exp() {
  local setting="$1"
  shift
  local save_dir="${OUT_ROOT}/${setting}"
  mkdir -p "${save_dir}"

  echo "[Run] setting=${setting}"
  python3 "${TRAIN_ENTRY}" \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --save_dir "${save_dir}" \
    2>&1 | tee "${save_dir}/train.log"
}

# Threshold ablation: keep NMF prototype + NMF pseudo unchanged, only vary the
# threshold policy (adaptive vs. a sweep of fixed thresholds).

# 0) Dynamic (adaptive) threshold baseline (no --fixed_tau)
run_exp dynamic_tau \
  --use_nmf_pseudo \
  --proto_init nmf

# 1) Fixed-threshold sweeps
run_exp fixed_tau_070 \
  --use_nmf_pseudo \
  --proto_init nmf \
  --fixed_tau 0.70

run_exp fixed_tau_075 \
  --use_nmf_pseudo \
  --proto_init nmf \
  --fixed_tau 0.75

run_exp fixed_tau_080 \
  --use_nmf_pseudo \
  --proto_init nmf \
  --fixed_tau 0.80

run_exp fixed_tau_085 \
  --use_nmf_pseudo \
  --proto_init nmf \
  --fixed_tau 0.85

run_exp fixed_tau_090 \
  --use_nmf_pseudo \
  --proto_init nmf \
  --fixed_tau 0.90

python3 "${PROJ_ROOT}/scripts/collect_ablation_results.py" \
  --root_dir "${OUT_ROOT}" \
  --out_csv "${OUT_ROOT}/ablation_results.csv"


echo "[Done] Ablation runs finished. Summary: ${OUT_ROOT}/ablation_results.csv"
