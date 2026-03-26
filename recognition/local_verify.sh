#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [ABS_CKPT_PATH_OR_DIR] [BACKBONE_TYPE] [GPU_IDS|auto] [BATCH_SIZE] [DATA_ROOT]"
    echo "No arguments -> use default values in this script."
    echo "Example(custom): $0 /nfs/zzzhong/codes/exp/TFace/recognition/ckpt/Backbone_Epoch_20_checkpoint.pth IR_50 auto 1024 /nfs/zzzhong/codes/exp/TFace/dataset/val_data"
    echo "Example(dir): $0 /nfs/zzzhong/codes/exp/TFace/recognition/ckpt IR_50 auto 1024 /nfs/zzzhong/codes/exp/TFace/dataset/val_data"
    exit 0
fi

# Default values. You can edit these once and run directly.
DEFAULT_CKPT_PATH="/nfs/zzzhong/codes/exp/TFace/recognition/ckpt/high_fq_baseline"
DEFAULT_BACKBONE="IR_50"
DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
DEFAULT_BATCH_SIZE="512"
DEFAULT_DATA_ROOT="/nfs/zzzhong/codes/exp/TFace/dataset/val_data"
DEFAULT_EMBEDDING_SIZE="512"
DEFAULT_PREPROCESS_MODE="${PREPROCESS_MODE:-high_freq}" # high_freq ｜ rgb 
DEFAULT_FREQ_KEEP_CHANNELS="${FREQ_KEEP_CHANNELS:-}"
DEFAULT_INPUT_CHANNELS="${INPUT_CHANNELS:-162}" # 162 for high_freq, 3 for rgb

CKPT_TARGET="${1:-${DEFAULT_CKPT_PATH}}"
BACKBONE="${2:-${DEFAULT_BACKBONE}}"
GPU_IDS="${3:-${DEFAULT_GPU_IDS}}"
BATCH_SIZE="${4:-${DEFAULT_BATCH_SIZE}}"
DATA_ROOT="${5:-${DEFAULT_DATA_ROOT}}"
EMBEDDING_SIZE="${EMBEDDING_SIZE:-${DEFAULT_EMBEDDING_SIZE}}"
PREPROCESS_MODE="${PREPROCESS_MODE:-${DEFAULT_PREPROCESS_MODE}}"
FREQ_KEEP_CHANNELS="${FREQ_KEEP_CHANNELS:-${DEFAULT_FREQ_KEEP_CHANNELS}}"
INPUT_CHANNELS="${INPUT_CHANNELS:-${DEFAULT_INPUT_CHANNELS}}"

if [[ "${CKPT_TARGET}" != /* ]]; then
    echo "Error: ckpt path must be an absolute path, got: ${CKPT_TARGET}"
    exit 1
fi

if [[ ! -e "${CKPT_TARGET}" ]]; then
    echo "Error: ckpt target not found: ${CKPT_TARGET}"
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
echo "ckpt_target    : ${CKPT_TARGET}"
echo "backbone       : ${BACKBONE}"
echo "gpu_ids(phys)  : ${GPU_IDS}"
echo "gpu_ids(logic) : ${LOGICAL_GPU_IDS}"
echo "batch_size     : ${BATCH_SIZE}"
echo "embedding_size : ${EMBEDDING_SIZE}"
echo "data_root      : ${DATA_ROOT}"
echo "preprocess     : ${PREPROCESS_MODE}"
echo "freq_keep_chs  : ${FREQ_KEEP_CHANNELS}"
echo "input_channels : ${INPUT_CHANNELS}"
echo "========================================="

run_single_verify() {
    local ckpt_path="$1"
    local results_json="$2"

    VERIFY_CMD=(
        python -u test/verification.py
        --ckpt_path "${ckpt_path}"
        --backbone "${BACKBONE}"
        --gpu_ids "${LOGICAL_GPU_IDS}"
        --batch_size "${BATCH_SIZE}"
        --data_root "${DATA_ROOT}"
        --embedding_size "${EMBEDDING_SIZE}"
        --preprocess_mode "${PREPROCESS_MODE}"
        --results_json "${results_json}"
    )
    if [[ -n "${FREQ_KEEP_CHANNELS}" ]]; then
        VERIFY_CMD+=(--freq_keep_channels "${FREQ_KEEP_CHANNELS}")
    fi
    if [[ -n "${INPUT_CHANNELS}" ]]; then
        VERIFY_CMD+=(--input_channels "${INPUT_CHANNELS}")
    fi
    "${VERIFY_CMD[@]}"
}

if [[ -f "${CKPT_TARGET}" ]]; then
    TMP_JSON="$(mktemp)"
    run_single_verify "${CKPT_TARGET}" "${TMP_JSON}"
    rm -f "${TMP_JSON}"
    exit 0
fi

RESULT_ROOT="${RESULT_ROOT:-${CKPT_TARGET}/verify_reports}"
RAW_LOG_DIR="${RESULT_ROOT}/raw_logs"
JSON_DIR="${RESULT_ROOT}/json"
mkdir -p "${RAW_LOG_DIR}" "${JSON_DIR}"

TIMESTAMP="$(date +%F-%H-%M-%S)"
REPORT_MD="${REPORT_MD:-${RESULT_ROOT}/verification_${TIMESTAMP}.md}"

mapfile -t CKPT_FILES < <(find "${CKPT_TARGET}" -maxdepth 1 -type f -name 'Backbone_Epoch_*_checkpoint.pth' | sort -V)
if [[ "${#CKPT_FILES[@]}" -eq 0 ]]; then
    echo "Error: no Backbone_Epoch_*_checkpoint.pth files found under ${CKPT_TARGET}"
    exit 1
fi

echo "# Verification Report" > "${REPORT_MD}"
echo "" >> "${REPORT_MD}"
echo "- CKPT directory: \`${CKPT_TARGET}\`" >> "${REPORT_MD}"
echo "- Backbone: \`${BACKBONE}\`" >> "${REPORT_MD}"
echo "- Preprocess: \`${PREPROCESS_MODE}\`" >> "${REPORT_MD}"
echo "- Generated at: \`$(date '+%F %T')\`" >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"
echo "| Epoch | Checkpoint | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW |" >> "${REPORT_MD}"
echo "| --- | --- | ---: | ---: | ---: | ---: | ---: |" >> "${REPORT_MD}"

for ckpt_path in "${CKPT_FILES[@]}"; do
    ckpt_name="$(basename "${ckpt_path}")"
    epoch="$(sed -n 's/^Backbone_Epoch_\([0-9]\+\)_checkpoint\.pth$/\1/p' <<< "${ckpt_name}")"
    if [[ -z "${epoch}" ]]; then
        epoch="-"
    fi

    json_path="${JSON_DIR}/${ckpt_name%.pth}.json"
    log_path="${RAW_LOG_DIR}/${ckpt_name%.pth}.log"

    echo ""
    echo ">>> Verifying ${ckpt_name}"
    run_single_verify "${ckpt_path}" "${json_path}" 2>&1 | tee "${log_path}"

    row="$(python - "${json_path}" "${epoch}" "${ckpt_name}" <<'PY'
import json
import sys

json_path, epoch, ckpt_name = sys.argv[1:4]
with open(json_path, "r") as f:
    data = json.load(f)
metrics = data["metrics"]
def fmt(key):
    return f'{metrics[key]["acc"]:.6f}'
print(f'| {epoch} | `{ckpt_name}` | {fmt("lfw")} | {fmt("cfp_fp")} | {fmt("agedb_30")} | {fmt("calfw")} | {fmt("cplfw")} |')
PY
)"
    echo "${row}" >> "${REPORT_MD}"
done

echo ""
echo "Markdown report saved to: ${REPORT_MD}"
