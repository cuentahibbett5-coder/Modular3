#!/bin/bash
#SBATCH --job-name=train_weighted_pro
#SBATCH --output=/home/fer/fer/Modular3/logs/train_weighted_pro_%j.log
#SBATCH --error=/home/fer/fer/Modular3/logs/train_weighted_pro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

# Create logs directory
mkdir -p /home/fer/fer/Modular3/logs

# Activate environment
source /home/fer/fer/Modular3/.venv/bin/activate

# Set environment for ROCM (if needed)
export HIP_VISIBLE_DEVICES=0

# Run training
echo "=========================================="
echo "Starting Weighted Pro Training"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $HIP_VISIBLE_DEVICES"
echo "=========================================="
echo ""

cd /home/fer/fer/Modular3
python3 train_weighted_pro.py

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
