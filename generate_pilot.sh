#!/bin/bash

# =================== CONFIGURACIÓN ===================
# Directorio base para el dataset
BASE_DIR="dataset_pilot"

# Archivo de Phase Space (Input)
PHSP_FILE="data/IAEA/Salida_Varian_OpenGate_mm.root"

# Script de simulación
SCRIPT_SIM="simulations/dose_simulation.py"

# Número de pares (train + val)
N_TRAIN=20
N_VAL=5
N_TOTAL=25

# Partículas por nivel (4 niveles de entrada + 1 target)
N_INPUT_1M=1000000
N_INPUT_2M=2000000
N_INPUT_5M=5000000
N_INPUT_10M=10000000
N_TARGET=29400000    # Todo el PHSP (~29.4M) - SEED ÚNICA

# Archivo de salida con la lista de tareas
OUTPUT_TASK_FILE="pilot_tasks.txt"
# =====================================================

# Limpiar archivo de tareas previo
> "$OUTPUT_TASK_FILE"

echo "Generando lista de tareas en $OUTPUT_TASK_FILE..."

# Función helper para escribir comando
add_task() {
    local out_dir=$1
    local n_part=$2
    local seed=$3

    local cmd="if [ ! -f ${out_dir}/dose_edep.mhd ]; then python ${SCRIPT_SIM} --input ${PHSP_FILE} --output ${out_dir} --n-particles ${n_part} --threads 1 --seed ${seed}; fi"

    echo "$cmd" >> "$OUTPUT_TASK_FILE"
}

# 1. Generar TARGETS MÚLTIPLES (Ground Truth con seeds diferentes)
# Un target cada 5 pares para que la red vea variabilidad natural
echo "Generando 5 targets únicos (29.4M eventos cada uno)..."
TARGET_SEEDS=(12345 54321 99999 11111 77777)
for i in "${!TARGET_SEEDS[@]}"; do
    target_idx=$((i + 1))
    add_task "${BASE_DIR}/target_${target_idx}" "$N_TARGET" "${TARGET_SEEDS[$i]}"
done

# Contadores
id_counter=1
target_idx=1

# 2. Generar Pares (Train)
echo "Generando $N_TRAIN pares de entrenamiento..."
for i in $(seq 1 $N_TRAIN); do
    pair_name=$(printf "pair_%03d" $id_counter)
    base_path="${BASE_DIR}/train/${pair_name}"

    # Seeds únicos por par: ID*1000 + nivel
    seed_1M=$((id_counter * 1000 + 1))
    seed_2M=$((id_counter * 1000 + 2))
    seed_5M=$((id_counter * 1000 + 5))
    seed_10M=$((id_counter * 1000 + 10))

    add_task "${base_path}/input_1M"  "$N_INPUT_1M"  "$seed_1M"
    add_task "${base_path}/input_2M"  "$N_INPUT_2M"  "$seed_2M"
    add_task "${base_path}/input_5M"  "$N_INPUT_5M"  "$seed_5M"
    add_task "${base_path}/input_10M" "$N_INPUT_10M" "$seed_10M"

    ((id_counter++))
    
    # Cambiar target cada 5 pares
    if (( i % 5 == 0 && i < N_TRAIN )); then
        ((target_idx++))
    fi
done

# 3. Generar Pares (Validation)
echo "Generando $N_VAL pares de validación..."
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
N_INPUTS=$((N_TOTAL * 4))
N_TARGETS=5
echo "============================================="
echo "✅ Completado."
echo "   Targets: $N_TARGETS sims (29.4M cada uno, seeds diferentes)"
echo "   Train: $N_TRAIN pares × 4 niveles = $((N_TRAIN * 4)) sims"
echo "   Val:   $N_VAL pares × 4 niveles = $((N_VAL * 4)) sims"
echo "   Total: $TOTAL tareas ($N_TARGETS targets + $N_INPUTS inputs)"
echo "   Archivo: $OUTPUT_TASK_FILE"
echo "============================================="
echo ""
echo "Distribución de jobs:"
echo "   1M:    $N_TOTAL sims × ~1.2 min   = ~$((N_TOTAL * 1)) min CPU"
echo "   2M:    $N_TOTAL sims × ~2.4 min   = ~$((N_TOTAL * 2)) min CPU"
echo "   5M:    $N_TOTAL sims × ~6 min     = ~$((N_TOTAL * 6)) min CPU"
echo "   10M:   $N_TOTAL sims × ~12 min    = ~$((N_TOTAL * 12)) min CPU"
echo "   29.4M: $N_TARGETS sims × ~35 min  = ~$((N_TARGETS * 35)) min CPU"
echo "   Total CPU time: ~$((N_TOTAL * 21 + N_TARGETS * 35)) min (~$((N_TOTAL * 21 / 60 + N_TARGETS * 35 / 60)) hrs)"
echo ""
echo "Para lanzar en el cluster:"
echo "   sbatch --array=1-${TOTAL}%50 run_pilot_slurm.sh"
