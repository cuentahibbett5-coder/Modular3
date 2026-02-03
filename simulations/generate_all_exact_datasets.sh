#!/bin/bash
# Genera todos los datasets necesarios para entrenamiento de red de denoising
# Usa direcciones reales del phase space IAEA

set -e

PHSP="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.npz"
OUTPUT_DIR="results/iaea_exact"
PHANTOM_SIZE=20
OFFSET=2
THREADS=1

echo "=========================================="
echo "GENERACIÓN DE DATASETS - Phase Space IAEA"
echo "=========================================="
echo ""

# Activar entorno
source .venv/bin/activate

# Datasets con diferentes estadísticas
# Input ruidosos: 100k, 500k, 1M, 2M
# Ground truth limpio: 10M, full (29M)

declare -a STATS=("100000" "500000" "1000000" "2000000" "10000000" "29406825")
declare -a LABELS=("100k" "500k" "1M" "2M" "10M" "full")

for i in "${!STATS[@]}"; do
    N="${STATS[$i]}"
    LABEL="${LABELS[$i]}"
    
    echo ""
    echo "=========================================="
    echo "Dataset: $LABEL ($N partículas)"
    echo "=========================================="
    
    OUTPUT="$OUTPUT_DIR/dose_$LABEL"
    
    if [ -f "$OUTPUT/dose_dose.mhd" ]; then
        echo "⚠️  Ya existe, saltando..."
        continue
    fi
    
    START=$(date +%s)
    
    python simulations/dose_exact_phsp.py \
        --phsp "$PHSP" \
        --output "$OUTPUT" \
        --particles "$N" \
        --phantom-size "$PHANTOM_SIZE" \
        --offset "$OFFSET" \
        --threads "$THREADS" \
        --seed 42
    
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    echo ""
    echo "✅ Completado en ${ELAPSED}s"
    echo ""
done

echo ""
echo "=========================================="
echo "✅ TODOS LOS DATASETS GENERADOS"
echo "=========================================="
echo ""
echo "Datasets disponibles en: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Próximo paso: Generar visualizaciones"
echo "  python simulations/visualize_iaea_datasets.py --datasets-dir $OUTPUT_DIR"
