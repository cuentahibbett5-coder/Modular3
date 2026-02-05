#!/bin/bash

# =================== CONFIGURACIÓN (EDITAR AQUÍ) ===================
BASE_DIR="dataset_pilot"
PHSP_FILE="data/IAEA/Salida_Varian_OpenGate_mm.root"
SCRIPT_SIM="simulations/dose_simulation.py"

# Estructura del dataset
N_TRAIN=20                  # Pares de entrenamiento
N_VAL=5                     # Pares de validación
N_TARGET=5                  # Número de targets (ground truth)
TARGET_SEEDS=(12345 54321 99999 11111 77777)

# Niveles de entrada (ruido) - cuántas partículas
N_INPUT_1M=1000000
N_INPUT_2M=2000000
N_INPUT_5M=5000000
N_INPUT_10M=10000000
N_FULL_PHSP=29400000        # Todo el PHSP para targets

# SLURM (modificar según tu cluster)
SLURM_MEM="4G"
SLURM_TIME="60:00"
SLURM_PARALLEL=50           # Máximo jobs simultáneos

OUTPUT_TASK_FILE="pilot_tasks.txt"
# ===================================================================

set -e

echo "🔧 Configuración:"
echo "   Base: $BASE_DIR"
echo "   PHSP: $PHSP_FILE"
echo "   Train: $N_TRAIN pares × 4 niveles = $((N_TRAIN * 4)) sims"
echo "   Val:   $N_VAL pares × 4 niveles = $((N_VAL * 4)) sims"
echo "   Targets: $N_TARGET sims (29.4M eventos cada uno)"
echo ""

# Limpiar
rm -f "$OUTPUT_TASK_FILE"
mkdir -p logs

# Función para agregar tarea
add_task() {
    local out_dir=$1
    local n_part=$2
    local seed=$3
    local cmd="if [ ! -f ${out_dir}/dose_edep.mhd ]; then python ${SCRIPT_SIM} --input ${PHSP_FILE} --output ${out_dir} --n-particles ${n_part} --threads 1 --seed ${seed}; fi"
    echo "$cmd" >> "$OUTPUT_TASK_FILE"
}

# Generar targets
echo "📊 Generando $N_TARGET targets..."
for i in "${!TARGET_SEEDS[@]}"; do
    target_idx=$((i + 1))
    add_task "${BASE_DIR}/target_${target_idx}" "$N_FULL_PHSP" "${TARGET_SEEDS[$i]}"
done

# Contadores
id_counter=1

# Generar pares (Train)
echo "📈 Generando $N_TRAIN pares de entrenamiento..."
for i in $(seq 1 $N_TRAIN); do
    pair_name=$(printf "pair_%03d" $id_counter)
    base_path="${BASE_DIR}/train/${pair_name}"

    seed_1M=$((id_counter * 1000 + 1))
    seed_2M=$((id_counter * 1000 + 2))
    seed_5M=$((id_counter * 1000 + 5))
    seed_10M=$((id_counter * 1000 + 10))

    add_task "${base_path}/input_1M"  "$N_INPUT_1M"  "$seed_1M"
    add_task "${base_path}/input_2M"  "$N_INPUT_2M"  "$seed_2M"
    add_task "${base_path}/input_5M"  "$N_INPUT_5M"  "$seed_5M"
    add_task "${base_path}/input_10M" "$N_INPUT_10M" "$seed_10M"

    ((id_counter++))
done

# Generar pares (Val)
echo "✅ Generando $N_VAL pares de validación..."
for i in $(seq 1 $N_VAL); do
    pair_name=$(printf "pair_%03d" $id_counter)
    base_path="${BASE_DIR}/val/${pair_name}"

    seed_1M=$((id_counter * 1000 + 1))
    seed_2M=$((id_counter * 1000 + 2))
    seed_5M=$((id_counter * 1000 + 5))
    seed_10M=$((id_counter * 1000 + 10))

    add_task "${base_path}/input_1M"  "$N_INPUT_1M"  "$seed_1M"
    add_task "${base_path}/input_2M"  "$N_INPUT_2M"  "$seed_2M"
    add_task "${base_path}/input_5M"  "$N_INPUT_5M"  "$seed_5M"
    add_task "${base_path}/input_10M" "$N_INPUT_10M" "$seed_10M"

    ((id_counter++))
done

TOTAL=$(wc -l < "$OUTPUT_TASK_FILE")

echo ""
echo "✅ Tareas generadas en $OUTPUT_TASK_FILE"
echo "   Total: $TOTAL simulaciones"
echo ""

# Lanzar en cluster
echo "🚀 Lanzando en cluster..."
sbatch \
    --array=1-${TOTAL}%${SLURM_PARALLEL} \
    --mem=${SLURM_MEM} \
    --time=${SLURM_TIME} \
    --job-name=pilot_sim \
    --output=logs/job_%a.log \
    --error=logs/job_%a.err \
    --wrap="COMMAND=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${OUTPUT_TASK_FILE}); eval \"\$COMMAND\""

echo "✅ Job array lanzado!"
echo "   Ver estado: squeue -j <JOBID>"
echo "   Logs en: logs/"
