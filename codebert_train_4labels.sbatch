#!/bin/bash
#SBATCH --job-name=CODEBERT4
#SBATCH --account=ml-vuln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-low
#SBATCH --constraint=gpu80g
#SBATCH --time=20-00:00:00
#SBATCH --mem=80G
#SBATCH --output=%j-codebert-4labels.out
#SBATCH --error=%j-codebert-4labels.err

set -euo pipefail

module load cuda
module load python/3.11.7-gcc12.1.0-x86_64

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV_DIR="${PROJECT_DIR}/env"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data4}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models/codebert-4labels}"

cd "${PROJECT_DIR}"

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    echo "Virtual environment not found at ${ENV_DIR}" >&2
    echo "Create it and install requirements before submitting the job." >&2
    exit 1
fi

if [[ ! -f "${DATA_DIR}/train.csv" || \
      ! -f "${DATA_DIR}/validation.csv" || \
      ! -f "${DATA_DIR}/test.csv" ]]; then
    echo "Four-label train, validation, or test split is missing from ${DATA_DIR}." >&2
    echo "Run curate_dataset_4labels.py before submitting the training job." >&2
    exit 1
fi

source "${ENV_DIR}/bin/activate"

if [[ "${VIRTUAL_ENV:-}" != "${ENV_DIR}" ]]; then
    echo "Failed to activate ${ENV_DIR}" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Dataset: ${DATA_DIR}"
echo "Model output: ${MODEL_DIR}"
echo "Python: $(command -v python)"
python --version
nvidia-smi

srun python -u codebert_train.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${MODEL_DIR}" \
    --train-limit 0 \
    "$@"
