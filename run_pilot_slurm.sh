#!/bin/bash

# =================== SLURM DIRECTIVES ===================
#SBATCH --array=1-105%50
#SBATCH --mem=4G
#SBATCH --time=60:00
#SBATCH --job-name=pilot_sim
#SBATCH --output=logs/job_%a.log
#SBATCH --error=logs/job_%a.err
# ========================================================

# Cambiar al directorio del script
cd "$(dirname "$0")"

# Crear directorio de logs si no existe
mkdir -p logs

# Leer la línea correspondiente del archivo de tareas
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pilot_tasks.txt)

# Ejecutar el comando
eval "$COMMAND"
