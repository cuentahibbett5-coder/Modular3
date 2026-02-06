#!/bin/bash
#SBATCH --job-name=validate_gt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/validate_gt_%j.log
#SBATCH --error=logs/validate_gt_%j.err

set -e

cd /lustre/home/acastaneda/Fernando/Modular3

echo "=========================================="
echo "VALIDACIÓN DE GROUND TRUTHS"
echo "=========================================="
echo "Fecha: $(date)"
echo "Node: $(hostname)"
echo ""

# Cargar módulos
module load rocm/5.7 python/3.10

# Activar venv
source venv/bin/activate

# Ejecutar validación
python validate_gt_cluster.py 2>&1 | tee validation_results.txt

echo ""
echo "=========================================="
echo "✅ Validación completada"
echo "Resultados guardados en validation_results.txt"
echo "Gráfico en: gt_validation_all_pdds.png"
echo "=========================================="
