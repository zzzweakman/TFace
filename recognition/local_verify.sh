#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [ABS_CKPT_PATH] [BACKBONE_TYPE] [GPU_IDS|auto] [BATCH_SIZE] [DATA_ROOT]"
    echo "No arguments -> use default values in this script."
    echo "Example(custom): $0 /nfs/zzzhong/codes/exp/TFace/recognition/ckpt/Backbone_Epoch_20_checkpoint.pth IR_50 auto 1024 /nfs/zzzhong/codes/exp/TFace/dataset/val_data"
    exit 0
fi

# Default values. You can edit these once and run directly.
DEFAULT_CKPT_PATH="/nfs/zzzhong/codes/exp/TFace/recognition/ckpt/Backbone_Epoch_18_checkpoint.pth"
DEFAULT_BACKBONE="IR_50"
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_BATCH_SIZE="16384"
DEFAULT_DATA_ROOT="/nfs/zzzhong/codes/exp/TFace/dataset/val_data"
DEFAULT_EMBEDDING_SIZE="512"

CKPT_PATH="${1:-${DEFAULT_CKPT_PATH}}"
BACKBONE="${2:-${DEFAULT_BACKBONE}}"
GPU_IDS="${3:-${DEFAULT_GPU_IDS}}"
BATCH_SIZE="${4:-${DEFAULT_BATCH_SIZE}}"
DATA_ROOT="${5:-${DEFAULT_DATA_ROOT}}"
EMBEDDING_SIZE="${EMBEDDING_SIZE:-${DEFAULT_EMBEDDING_SIZE}}"

if [[ "${CKPT_PATH}" != /* ]]; then
    echo "Error: ckpt path must be an absolute path, got: ${CKPT_PATH}"
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Error: ckpt file not found: ${CKPT_PATH}"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Error: data_root not found: ${DATA_ROOT}"
    exit 1
fi

REQUIRED_BINS=("lfw.bin" "cfp_fp.bin" "agedb_30.bin" "calfw.bin" "cplfw.bin")
for bin_name in "${REQUIRED_BINS[@]}"; do
    if [[ ! -f "${DATA_ROOT}/${bin_name}" ]]; then
        echo "Error: missing validation file: ${DATA_ROOT}/${bin_name}"
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${GPU_IDS}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d '[:space:]')"
    else
        GPU_COUNT="1"
    fi
    if [[ "${GPU_COUNT}" -le 0 ]]; then
        GPU_COUNT="1"
    fi
    GPU_IDS="$(seq -s, 0 $((GPU_COUNT - 1)))"
fi

GPU_IDS="${GPU_IDS// /}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -le 0 ]]; then
    echo "Error: invalid GPU_IDS: ${GPU_IDS}"
    exit 1
fi

# Select physical GPUs through CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
# DataParallel device_ids are relative to visible GPUs.
LOGICAL_GPU_IDS="$(seq -s, 0 $((${#GPU_ARRAY[@]} - 1)))"

echo "========== Verification Config =========="
echo "ckpt_path      : ${CKPT_PATH}"
echo "backbone       : ${BACKBONE}"
echo "gpu_ids(phys)  : ${GPU_IDS}"
echo "gpu_ids(logic) : ${LOGICAL_GPU_IDS}"
echo "batch_size     : ${BATCH_SIZE}"
echo "embedding_size : ${EMBEDDING_SIZE}"
echo "data_root      : ${DATA_ROOT}"
echo "========================================="

python -u test/verification.py \
    --ckpt_path "${CKPT_PATH}" \
    --backbone "${BACKBONE}" \
    --gpu_ids "${LOGICAL_GPU_IDS}" \
    --batch_size "${BATCH_SIZE}" \
    --data_root "${DATA_ROOT}" \
    --embedding_size "${EMBEDDING_SIZE}"
