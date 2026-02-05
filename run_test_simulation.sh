#!/bin/bash

# TEST rápido: Solo 10k partículas para verificar que funciona
PHSP_FILE="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root"
OUTPUT_DIR="output/dose_maps/test_10k"

python3 simulations/dose_simulation.py \
    --input "$PHSP_FILE" \
    --output "$OUTPUT_DIR" \
    --n-particles 10000 \
    --threads 4 \
    --seed 42 \
    --spacing-xy 2.0 \
    --spacing-z 1.0 \
    --gap 50 \
    --dry-run

echo ""
echo "Para ejecutar sin --dry-run:"
echo "bash run_test_simulation.sh"
