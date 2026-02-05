#!/bin/bash
# =============================================================================
# Test local de simulación
# =============================================================================
#
# Uso:
#   ./run_local.sh [N_PARTICLES] [THREADS]
#
# Ejemplos:
#   ./run_local.sh              # 10000 partículas, 4 threads
#   ./run_local.sh 100000 8     # 100k partículas, 8 threads
#   ./run_local.sh --dry-run    # solo mostrar config
#
# =============================================================================

set -e

cd "$(dirname "$0")"

# Activar venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Parámetros
if [ "$1" == "--dry-run" ]; then
    DRY_RUN="--dry-run"
    N_PARTICLES=1000
    THREADS=1
else
    N_PARTICLES="${1:-10000}"
    THREADS="${2:-4}"
    DRY_RUN=""
fi

PHSP="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"
OUTPUT="output/test_local"

echo "=============================================="
echo "Test local de simulación"
echo "=============================================="
echo "  PHSP:       $PHSP"
echo "  Output:     $OUTPUT"
echo "  Particles:  $N_PARTICLES"
echo "  Threads:    $THREADS"
echo "=============================================="

python3 simulations/dose_simulation.py \
    --input "$PHSP" \
    --output "$OUTPUT" \
    --n-particles "$N_PARTICLES" \
    --threads "$THREADS" \
    --seed 12345 \
    $DRY_RUN

echo ""
echo "Done! Results in $OUTPUT/"
