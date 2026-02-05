#!/bin/bash
# =============================================================================
# Launcher para simulaciones en cluster SLURM
# =============================================================================
#
# Uso:
#   ./launch_simulations.sh [ARRAY_RANGE] [N_PARTICLES] [CONCURRENT]
#
# Ejemplos:
#   ./launch_simulations.sh 1-100 1000000 4    # 100 jobs, 1M partículas, 4 concurrent
#   ./launch_simulations.sh 1-25 10000000 2    # 25 jobs, 10M partículas, 2 concurrent
#   ./launch_simulations.sh 1-100              # defaults: 1M, 2 concurrent
#
# =============================================================================

set -e

# Parámetros
ARRAY_RANGE="${1:-1-100}"
N_PARTICLES="${2:-1000000}"
CONCURRENT="${3:-2}"

# Configuración cluster
VENV_PATH="/lustre/home/acastaneda/Fernando/Modular3/.venv"
THREADS=8
TIME="02:00:00"
MEM="32G"

# Directorios
PHSP_FILE="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"
OUTPUT_BASE="output"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Launching SLURM array job"
echo "=============================================="
echo "  Array:      ${ARRAY_RANGE}%${CONCURRENT}"
echo "  Particles:  ${N_PARTICLES}"
echo "  Threads:    ${THREADS}"
echo "  Time:       ${TIME}"
echo "  Memory:     ${MEM}"
echo "=============================================="

sbatch --job-name=dose_sim \
  --array=${ARRAY_RANGE}%${CONCURRENT} \
  --ntasks=1 \
  --cpus-per-task=${THREADS} \
  --mem=${MEM} \
  --time=${TIME} \
  --output=${LOG_DIR}/job_%A_%a.out \
  --error=${LOG_DIR}/job_%A_%a.err \
  --export=ALL,N_PARTICLES=${N_PARTICLES},THREADS=${THREADS},VENV_PATH=${VENV_PATH},PHSP_FILE=${PHSP_FILE},OUTPUT_BASE=${OUTPUT_BASE} \
  <<'SLURM_SCRIPT'
#!/bin/bash

# Activar entorno
if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
elif [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
fi

cd "$SLURM_SUBMIT_DIR" || exit 1

# ID del par (3 dígitos)
PAIR_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

# Seed único para cada job
SEED=$((10000 + SLURM_ARRAY_TASK_ID))

# Directorio de salida
OUT_DIR="${OUTPUT_BASE}/pair_${PAIR_ID}"

echo "========================================"
echo "Job $PAIR_ID - $(date)"
echo "========================================"

python3 simulations/dose_simulation.py \
  --input "$PHSP_FILE" \
  --output "$OUT_DIR" \
  --n-particles "$N_PARTICLES" \
  --threads "$THREADS" \
  --seed "$SEED" \
  --job-id "pair_${PAIR_ID}"

echo "Done: $OUT_DIR"
SLURM_SCRIPT

echo ""
echo "Submitted! Check with: squeue -u $USER"
echo "Logs in: ${LOG_DIR}/"
