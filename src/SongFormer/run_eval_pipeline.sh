#!/bin/bash
# SongFormBench Evaluation Pipeline
# Usage: bash run_eval_pipeline.sh [--use_mirror] [--gpu_num N] [--skip_download]
#        bash run_eval_pipeline.sh --local_bench_dir /path/to/SongFormBench [--gpu_num N]
#
# This script runs the full evaluation pipeline:
#   1. Download SongFormBench dataset from HuggingFace (or prepare from local)
#   2. Download pretrained model checkpoints
#   3. Run inference on each subset
#   4. Convert inference results to MSA TXT format
#   5. Run evaluation metrics
#   6. Summarize results into a single MD file

set -e

# ============== Activate conda environment ==============
CONDA_ENV_NAME="songformer"
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    # Initialize conda for non-interactive shell
    __conda_setup="$(conda shell.bash hook 2>/dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        echo "Error: conda not found. Please install conda or activate the '${CONDA_ENV_NAME}' environment manually."
        exit 1
    fi
    conda activate "$CONDA_ENV_NAME"
    echo "Activated conda environment: ${CONDA_ENV_NAME}"
else
    echo "Conda environment '${CONDA_ENV_NAME}' is already active."
fi

# ============== Configuration ==============
USE_MIRROR=""
GPU_NUM=1
THREADS_PER_GPU=1
SKIP_DOWNLOAD=false
SKIP_MODEL_DOWNLOAD=false
LOCAL_BENCH_DIR=""
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
        --skip_model_download)
            SKIP_MODEL_DOWNLOAD=true
            shift
            ;;
        --local_bench_dir)
            LOCAL_BENCH_DIR="$2"
            shift 2
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${SCRIPT_DIR}/../third_party:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============== Helper: Prepare audio.scp and GT from a SongFormBench directory ==============
prepare_from_local() {
    local BENCH_DIR="$1"
    for SUBSET in "${SUBSETS[@]}"; do
        AUDIO_SRC="${BENCH_DIR}/data/audios/${SUBSET}"
        LABEL_SRC="${BENCH_DIR}/data/labels/${SUBSET}"

        if [ ! -d "$AUDIO_SRC" ]; then
            echo "  Warning: ${AUDIO_SRC} not found, skipping subset ${SUBSET}"
            continue
        fi
        if [ ! -d "$LABEL_SRC" ]; then
            echo "  Warning: ${LABEL_SRC} not found, skipping subset ${SUBSET}"
            continue
        fi

        SUBSET_DIR="${DATA_DIR}/${SUBSET}"
        GT_DIR="${SUBSET_DIR}/gt"
        mkdir -p "${GT_DIR}"

        # Create audio.scp with absolute paths
        : > "${SUBSET_DIR}/audio.scp"
        shopt -s nullglob
        for f in "${AUDIO_SRC}"/*.mp3 "${AUDIO_SRC}"/*.wav "${AUDIO_SRC}"/*.flac; do
            echo "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" >> "${SUBSET_DIR}/audio.scp"
        done
        shopt -u nullglob

        # Symlink GT label files
        for f in "${LABEL_SRC}"/*.txt; do
            [ -e "$f" ] || continue
            target="${GT_DIR}/$(basename "$f")"
            if [ ! -e "$target" ]; then
                ln -s "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$target"
            fi
        done

        SAMPLE_COUNT=$(wc -l < "${SUBSET_DIR}/audio.scp" | tr -d ' ')
        echo "  Prepared ${SUBSET}: ${SAMPLE_COUNT} audio files, $(ls "${GT_DIR}" | wc -l | tr -d ' ') GT labels"
    done
}

echo "========================================"
echo "SongFormBench Evaluation Pipeline"
echo "========================================"
echo "GPU_NUM: ${GPU_NUM}"
echo "DATA_DIR: ${DATA_DIR}"
echo "LOCAL_BENCH_DIR: ${LOCAL_BENCH_DIR:-auto-detect}"
echo "USE_MIRROR: ${USE_MIRROR:-no}"
echo "========================================"

# ============== Step 1: Download / Prepare Dataset ==============
# Auto-detect local SongFormBench if not specified
if [ -z "$LOCAL_BENCH_DIR" ]; then
    DEFAULT_BENCH="${SCRIPT_DIR}/../../SongFormBench"
    if [ -d "$DEFAULT_BENCH/data/audios" ]; then
        LOCAL_BENCH_DIR="$DEFAULT_BENCH"
        echo ""
        echo "Auto-detected local SongFormBench at: ${DEFAULT_BENCH}"
    fi
fi

if [ -n "$LOCAL_BENCH_DIR" ]; then
    # Use locally downloaded SongFormBench
    LOCAL_BENCH_DIR="$(cd "$LOCAL_BENCH_DIR" && pwd)"
    echo ""
    echo "[Step 1/6] Preparing data from local SongFormBench: ${LOCAL_BENCH_DIR}"
    prepare_from_local "${LOCAL_BENCH_DIR}"

elif [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "[Step 1/6] Downloading SongFormBench dataset via huggingface-cli..."
    HF_DOWNLOAD_DIR="${DATA_DIR}/_hf_download"
    if [ -n "$USE_MIRROR" ]; then
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
    huggingface-cli download \
        ASLP-lab/SongFormBench \
        --repo-type dataset \
        --local-dir "${HF_DOWNLOAD_DIR}"

    # Prepare audio.scp and GT from downloaded repo
    prepare_from_local "${HF_DOWNLOAD_DIR}"
else
    echo ""
    echo "[Step 1/6] Skipping dataset download (--skip_download)"
fi

# ============== Step 2: Download Pretrained Models ==============
if [ "$SKIP_DOWNLOAD" = false ] && [ "$SKIP_MODEL_DOWNLOAD" = false ]; then
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

    # Step 3: Run inference (capture RTF stats)
    mkdir -p "${INFER_JSON_DIR}"
    RTF_JSON="${DATA_DIR}/${SUBSET}/rtf_stats.json"
    INFER_DONE_COUNT=$(find "${INFER_JSON_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${INFER_DONE_COUNT}" -ge "${SAMPLE_COUNT}" ] && [ "${SAMPLE_COUNT}" -gt 0 ]; then
        echo "[Step 3/6] Inference already complete for ${SUBSET} (${INFER_DONE_COUNT}/${SAMPLE_COUNT}), skipping."
    else
        echo "[Step 3/6] Running inference on ${SUBSET} (${INFER_DONE_COUNT}/${SAMPLE_COUNT} done)..."
        python infer/infer.py \
            -i "${AUDIO_SCP}" \
            -o "${INFER_JSON_DIR}" \
            --model SongFormer \
            --checkpoint SongFormer.safetensors \
            --config_path SongFormer.yaml \
            -gn "${GPU_NUM}" \
            -tn "${THREADS_PER_GPU}" \
            --rtf_output "${RTF_JSON}"
    fi

    # Step 4: Convert JSON to MSA TXT
    INFER_JSON_COUNT=$(find "${INFER_JSON_DIR}" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    INFER_TXT_COUNT=$(find "${INFER_TXT_DIR}" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
    if [ "${INFER_TXT_COUNT}" -ge "${INFER_JSON_COUNT}" ] && [ "${INFER_JSON_COUNT}" -gt 0 ]; then
        echo "[Step 4/6] TXT conversion already complete for ${SUBSET} (${INFER_TXT_COUNT}/${INFER_JSON_COUNT}), skipping."
    else
        echo "[Step 4/6] Converting inference results to MSA TXT format..."
        python utils/convert_res2msa_txt.py \
            --input_folder "${INFER_JSON_DIR}" \
            --output_folder "${INFER_TXT_DIR}"
    fi

    # Step 5: Run evaluation
    EVAL_CSV="${EVAL_DIR}/eval_infer_summary.csv"
    if [ -f "${EVAL_CSV}" ]; then
        echo "[Step 5/6] Evaluation already complete for ${SUBSET}, skipping."
    else
        echo "[Step 5/6] Running evaluation on ${SUBSET}..."
        mkdir -p "${EVAL_DIR}"
        python evaluation/eval_infer_results.py \
            --ann_dir "${GT_DIR}" \
            --est_dir "${INFER_TXT_DIR}" \
            --output_dir "${EVAL_DIR}"
    fi
done

# ============== Step 6: Summarize Results ==============
echo ""
echo "[Step 6/6] Summarizing evaluation results..."
# Collect RTF JSON paths
RTF_ARGS=""
for SUBSET in "${SUBSETS[@]}"; do
    RTF_JSON="${DATA_DIR}/${SUBSET}/rtf_stats.json"
    if [ -f "${RTF_JSON}" ]; then
        RTF_ARGS="${RTF_ARGS} --rtf_files ${SUBSET}:${RTF_JSON}"
    fi
done

python utils/summarize_results.py \
    --eval_base_dir "${EVAL_OUTPUT_BASE}" \
    --output_path "${SUMMARY_OUTPUT}" \
    ${RTF_ARGS}

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Summary: ${SUMMARY_OUTPUT}"
echo "========================================"
