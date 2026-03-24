#!/bin/bash
# SongFormBench Evaluation Pipeline
# Usage: bash run_eval_pipeline.sh [--use_mirror] [--gpu_num N] [--skip_download]
#
# This script runs the full evaluation pipeline:
#   1. Download SongFormBench dataset from HuggingFace
#   2. Download pretrained model checkpoints
#   3. Run inference on each subset
#   4. Convert inference results to MSA TXT format
#   5. Run evaluation metrics
#   6. Summarize results into a single MD file

set -e

# ============== Configuration ==============
USE_MIRROR=""
GPU_NUM=1
THREADS_PER_GPU=1
SKIP_DOWNLOAD=false
DATA_DIR="eval_results/SongFormBench"
EVAL_OUTPUT_BASE="eval_results/eval_output"
SUMMARY_OUTPUT="eval_results/evaluation_summary.md"
SUBSETS=("HarmonixSet" "CN")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_mirror)
            USE_MIRROR="--use_mirror"
            shift
            ;;
        --gpu_num)
            GPU_NUM="$2"
            shift 2
            ;;
        --threads_per_gpu)
            THREADS_PER_GPU="$2"
            shift 2
            ;;
        --skip_download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============== Environment ==============
export PYTHONPATH="${PWD}:../third_party:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "========================================"
echo "SongFormBench Evaluation Pipeline"
echo "========================================"
echo "GPU_NUM: ${GPU_NUM}"
echo "DATA_DIR: ${DATA_DIR}"
echo "USE_MIRROR: ${USE_MIRROR:-no}"
echo "========================================"

# ============== Step 1: Download Dataset ==============
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[Step 1/6] Downloading SongFormBench dataset..."
    python utils/download_songformbench.py \
        --output_dir "${DATA_DIR}" \
        ${USE_MIRROR}
else
    echo ""
    echo "[Step 1/6] Skipping dataset download (--skip_download)"
fi

# ============== Step 2: Download Pretrained Models ==============
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[Step 2/6] Downloading pretrained model checkpoints..."
    if [ -n "$USE_MIRROR" ]; then
        python -c "from utils.fetch_pretrained import download_all; download_all(use_mirror=True)"
    else
        python -c "from utils.fetch_pretrained import download_all; download_all(use_mirror=False)"
    fi
else
    echo ""
    echo "[Step 2/6] Skipping model download (--skip_download)"
fi

# ============== Steps 3-5: Per-subset Inference & Evaluation ==============
for SUBSET in "${SUBSETS[@]}"; do
    AUDIO_SCP="${DATA_DIR}/${SUBSET}/audio.scp"
    GT_DIR="${DATA_DIR}/${SUBSET}/gt"
    INFER_JSON_DIR="${DATA_DIR}/${SUBSET}/infer_json"
    INFER_TXT_DIR="${DATA_DIR}/${SUBSET}/infer_txt"
    EVAL_DIR="${EVAL_OUTPUT_BASE}/${SUBSET}"

    # Check if this subset exists
    if [ ! -f "${AUDIO_SCP}" ]; then
        echo ""
        echo "Warning: ${AUDIO_SCP} not found, skipping subset ${SUBSET}"
        continue
    fi

    SAMPLE_COUNT=$(wc -l < "${AUDIO_SCP}" | tr -d ' ')
    echo ""
    echo "========================================"
    echo "Processing subset: ${SUBSET} (${SAMPLE_COUNT} samples)"
    echo "========================================"

    # Step 3: Run inference
    echo "[Step 3/6] Running inference on ${SUBSET}..."
    mkdir -p "${INFER_JSON_DIR}"
    python infer/infer.py \
        -i "${AUDIO_SCP}" \
        -o "${INFER_JSON_DIR}" \
        --model SongFormer \
        --checkpoint SongFormer.safetensors \
        --config_path SongFormer.yaml \
        -gn "${GPU_NUM}" \
        -tn "${THREADS_PER_GPU}"

    # Step 4: Convert JSON to MSA TXT
    echo "[Step 4/6] Converting inference results to MSA TXT format..."
    python utils/convert_res2msa_txt.py \
        --input_folder "${INFER_JSON_DIR}" \
        --output_folder "${INFER_TXT_DIR}"

    # Step 5: Run evaluation
    echo "[Step 5/6] Running evaluation on ${SUBSET}..."
    mkdir -p "${EVAL_DIR}"
    python evaluation/eval_infer_results.py \
        --ann_dir "${GT_DIR}" \
        --est_dir "${INFER_TXT_DIR}" \
        --output_dir "${EVAL_DIR}"
done

# ============== Step 6: Summarize Results ==============
echo ""
echo "[Step 6/6] Summarizing evaluation results..."
python utils/summarize_results.py \
    --eval_base_dir "${EVAL_OUTPUT_BASE}" \
    --output_path "${SUMMARY_OUTPUT}"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Summary: ${SUMMARY_OUTPUT}"
echo "========================================"
