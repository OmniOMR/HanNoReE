#!/usr/bin/env bash
#SBATCH --job-name=hannoree-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Evaluates a trained model (SER of greedy decoding) on a pickled split of a dataset, by default the test split of
# OmniOMR, datasets/omniomr/samples.test.pickle.
#
# Settings (environment variables):
#   MODEL        the checkpoint (default: experiment_results/best.pt)
#   DATASET      the dataset: a subfolder name of DATASET_DIR, a dataset folder or a .pickle file (default: omniomr)
#   DATASET_DIR  the directory with the datasets as subfolders (default: datasets)
#   SPLIT        dev or test (default: test)
#   OUTPUT       the JSON with the results (default: <SPLIT>_<DATASET>.json next to the checkpoint)
#   BF16         0 evaluates in float32; by default the backbone runs in bfloat16 autocast, as in training with
#                --musvit-bf16
# Extra arguments are passed to `hannoree evaluate`, e.g.:
#   MODEL=experiment_results/best.pt DATASET=dolores sbatch scripts/evaluate.slurm --batch-size 64
#
# Run from the repository root (the paths are relative to it); logs/ must exist before submitting.

set -euo pipefail

MODEL="${MODEL:-experiment_results/best.pt}"
DATASET="${DATASET:-omniomr}"
DATASET_DIR="${DATASET_DIR:-datasets}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-$(dirname "${MODEL}")/${SPLIT}_$(basename "${DATASET}" .pickle).json}"
BF16="${BF16:-1}"

if [ ! -f "${MODEL}" ]; then
    echo "Model checkpoint not found: ${MODEL}" >&2
    exit 1
fi

BF16_ARGS=()
if [ "${BF16}" = "1" ]; then
    BF16_ARGS=(--bf16)
fi

source .venv/bin/activate

# Real-time line buffering in the logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ${SLURM_JOB_ID:-N/A} on $(hostname), $(date); model: ${MODEL}; dataset: ${DATASET} (${SPLIT})"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

hannoree evaluate \
    --model "${MODEL}" \
    --dataset-dir "${DATASET_DIR}" \
    --datasets "${DATASET}" \
    --split "${SPLIT}" \
    --output "${OUTPUT}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --device cuda \
    ${BF16_ARGS[@]+"${BF16_ARGS[@]}"} \
    "$@"

echo "Done at $(date). Results: ${OUTPUT}"
