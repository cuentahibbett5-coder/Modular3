#!/bin/bash
# run_complete_workflow.sh
# Script para ejecutar workflow completo del proyecto

set -e  # Exit on error

echo "=========================================="
echo "Workflow Completo - Proyecto Modular 3"
echo "=========================================="

# 1. Generar phase space del linac
echo ""
echo "[1/6] Generando phase space del linac..."
python simulations/linac_6mv.py \
    --energy 5.8 \
    --particles 1e8 \
    --output data/phase_space/linac_6mv_10x10.root

# 2. Generar dataset de entrenamiento
echo ""
echo "[2/6] Generando dataset de entrenamiento..."
python data/dataset_generator.py \
    --output data/training/ \
    --phantoms water bone lung \
    --fields 5 10 15 \
    --samples-per-config 20

# 3. Entrenar modelo MCDNet
echo ""
echo "[3/6] Entrenando modelo MCDNet..."
python models/training.py \
    --data-dir data/training/ \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda

# 4. Aplicar inferencia a datos de validación
echo ""
echo "[4/6] Aplicando inferencia..."
python models/inference.py \
    --model models/checkpoints/mcdnet_best.pth \
    --input data/validation/sample_001_low_dose.mhd \
    --output results/sample_001_denoised.mhd

# 5. Calcular gamma index
echo ""
echo "[5/6] Calculando Gamma Index..."
python analysis/gamma_index.py \
    --reference data/validation/sample_001_clean_dose.mhd \
    --evaluated results/sample_001_denoised.mhd \
    --criteria 3%/3mm \
    --output results/gamma_analysis/

# 6. Generar visualizaciones
echo ""
echo "[6/6] Generando visualizaciones..."
python analysis/visualization.py \
    --dose results/sample_001_denoised.mhd \
    --plot-type pdd \
    --output results/figures/pdd_curve.png

python analysis/visualization.py \
    --dose results/sample_001_denoised.mhd \
    --plot-type profile \
    --output results/figures/beam_profile.png

echo ""
echo "=========================================="
echo "✓ Workflow completado exitosamente"
echo "=========================================="
echo ""
echo "Resultados guardados en: results/"
echo "  - Dosis denoised: results/sample_001_denoised.mhd"
echo "  - Gamma analysis: results/gamma_analysis/"
echo "  - Figuras: results/figures/"
