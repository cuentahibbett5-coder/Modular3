#!/bin/bash
#SBATCH --job-name=train_weighted
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi250x:1
#SBATCH --output=logs/train_weighted_%j.log
#SBATCH --error=logs/train_weighted_%j.err

set -e

# Setup
echo "================================"
echo "Training con Máscara Ponderada"
echo "================================"
date
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

cd /lustre/home/acastaneda/Fernando/Modular3

# Activate venv
source .venv/bin/activate

# Set ROCm
export ROCM_HOME=/opt/rocm
export HIP_PLATFORM=amd
export HSA_OVERRIDE_GFX_VERSION=90a:908

# Disable MIOpen
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

echo "Starting weighted training..."
python train_weighted.py

echo ""
echo "================================"
echo "Training completado"
date
