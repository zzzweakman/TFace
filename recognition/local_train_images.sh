#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${LOG_DIR:-logs}"
CKPT_DIR="${CKPT_DIR:-ckpt}"
TB_DIR="${TB_DIR:-tensorboard}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${TB_DIR}"

# Default to all 8 GPUs, override if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Infer local process count from visible GPU list when not explicitly set.
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
DEFAULT_NPROC="${#GPU_IDS[@]}"
if [[ "${DEFAULT_NPROC}" -le 0 ]]; then
    DEFAULT_NPROC=1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train.py}"

# RUN_IN_BACKGROUND=1 -> nohup background run, otherwise run in foreground.
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/images_$(date +%F-%H-%M-%S).log}"

if command -v torchrun >/dev/null 2>&1; then
    LAUNCH_CMD=(
        torchrun
        --nproc_per_node="${NPROC_PER_NODE}"
        --nnodes="${NNODES}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
        "${TRAIN_ENTRY}"
    )
else
    LAUNCH_CMD=(
        python -u -m torch.distributed.launch
        --nproc_per_node="${NPROC_PER_NODE}"
        --nnodes="${NNODES}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
        "${TRAIN_ENTRY}"
    )
fi

echo "Working directory: ${SCRIPT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}, NNODES=${NNODES}, NODE_RANK=${NODE_RANK}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND}"
echo "Log file: ${LOG_FILE}"
echo "Launch command: ${LAUNCH_CMD[*]}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
    nohup "${LAUNCH_CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo "Background training started. PID=$!"
    echo "Tail log with: tail -f ${LOG_FILE}"
else
    "${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
