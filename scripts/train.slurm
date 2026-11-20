#!/usr/bin/env bash
#SBATCH --job-name=hannoree
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Trains the HanNoReE model as it was trained for release. The model settings are the defaults of `hannoree train`
# (see the README); this script adds the datasets, the hardware settings and resuming.
#
# The backbone is downloaded from Hugging Face (gated: accept the terms on the model page and export HF_TOKEN).
# If compute nodes have no internet access, download it on a login node first:
#   hf download PRAIG/musvit
#
# OUTPUT_DIR selects the output directory (default: experiment_results); extra arguments are passed to
# `hannoree train`, e.g.: sbatch scripts/train.slurm --epochs 400

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-experiment_results}"

# Keep a copy of this script and its arguments with the results, so the run can be replayed. Under SLURM, $0 is the
# copy that was submitted, so later edits of the original do not leak in.
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/train.slurm"
if [ $# -gt 0 ]; then
    printf '%q ' "$@" > "${OUTPUT_DIR}/train.args"
    echo >> "${OUTPUT_DIR}/train.args"
fi

source .venv/bin/activate

# Real-time line buffering in the logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ${SLURM_JOB_ID:-N/A} on $(hostname), $(date); output: ${OUTPUT_DIR}"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# --grad-checkpointing is left off: at 96 x 1024 px (384 patches per stave), fine-tuning fits in memory without it,
# and it would recompute the backbone in every backward pass (~30% slower). Add it if the GPU runs out of memory.
hannoree train \
    --dataset-dir datasets \
    --output-dir "${OUTPUT_DIR}" \
    --train dolores olimpicsynth omniomr \
    --dev omniomr \
    --test dolores olimpicscan olimpicsynth omniomr \
    --musvit-bf16 \
    --resume auto \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --device cuda \
    "$@"

echo "Done at $(date). Results: ${OUTPUT_DIR}/results.json, ${OUTPUT_DIR}/results_table.md"
