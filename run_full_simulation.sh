#!/bin/bash

# Simulación con el dataset completo IAEA convertido a ROOT (29.4M partículas)
# Puedes ajustar los parámetros según necesites

PHSP_FILE="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root"
OUTPUT_DIR="output/dose_maps/full_phsp"
N_PARTICLES=1000000    # Comenzar con 1M para prueba
THREADS=8
SEED=42

echo "=================================================="
echo "Ejecutando simulación con PHSP completo"
echo "=================================================="
echo "PHSP File: $PHSP_FILE"
echo "Partículas: $N_PARTICLES"
echo "Output: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo ""

python3 simulations/dose_simulation.py \
    --input "$PHSP_FILE" \
    --output "$OUTPUT_DIR" \
    --n-particles "$N_PARTICLES" \
    --threads "$THREADS" \
    --seed "$SEED" \
    --spacing-xy 2.0 \
    --spacing-z 1.0 \
    --gap 50

echo ""
echo "✅ Simulación completada"
echo "Resultados en: $OUTPUT_DIR"
