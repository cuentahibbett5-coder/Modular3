#!/bin/bash

# =================== SLURM DIRECTIVES ===================
#SBATCH --job-name=denoise_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# ========================================================

set -e

cd "$(dirname "$0")"
mkdir -p logs runs

# Ruta del dataset en el cluster
DATA_ROOT="/lustre/home/acastaneda/Fernando/Modular3/dataset_pilot"

# Activar entorno virtual
source .venv/bin/activate

python train.py \
  --data-root "$DATA_ROOT" \
  --output-dir runs/denoising \
  --epochs 50 \
  --batch-size 2 \
  --patch-size 64,64,64 \
  --lr 1e-4 \
  --num-workers 4 \
  --cache-size 8 \
  --device auto \
  --amp \
  --val-every 1 \
  --save-every 5
