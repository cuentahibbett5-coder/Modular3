import os
import random

# Configuración
BASE_DIR = "dataset_pilot"
PHSP_FILE = "data/IAEA/Salida_Varian_OpenGate_mm.root"
SCRIPT_SIM = "simulations/dose_simulation.py"

# Definición del dataset
N_TRAIN = 40
N_VAL = 10
INPUT_LEVELS = {"input_1M": 1000000, "input_5M": 5000000}

# Comandos para el script
commands = []

def create_job(output_dir, n_particles, seed):
    """Genera el comando para una simulación"""
    cmd = (
        f"python {SCRIPT_SIM} "
        f"--input {PHSP_FILE} "
        f"--output {output_dir} "
        f"--n-particles {n_particles} "
        f"--threads 1 "
        f"--seed {seed} "
    )
    # Comprobación para no re-ejecutar si ya existe
    check_cmd = f"if [ ! -f {output_dir}/dose_edep.mhd ]; then {cmd}; fi"
    return check_cmd

# 1. Generar Target (Ground Truth) - ~30M events (todo el PHSP)
# Nota: Si el archivo tiene 29.4M, pedir 30M usará todos y reciclará un poco al final,
# o simplemente dejará de leer si el script controla el EOF.
# Asumiendo que 29400000 es el límite safe.
target_dir = os.path.join(BASE_DIR, "target_full")
commands.append(f"# --- TARGET FULL (Ground Truth) ---")
commands.append(create_job(target_dir, 29400000, 12345))

# 2. Generar Pares (Train + Val)
id_counter = 1
for stage, n_samples in [("train", N_TRAIN), ("val", N_VAL)]:
    for i in range(n_samples):
        pair_name = f"pair_{id_counter:03d}"
        
        for input_name, counts in INPUT_LEVELS.items():
            out_path = os.path.join(BASE_DIR, stage, pair_name, input_name)
            
            # Seed única basada en ID y tipo
            # ID=1 -> 1M: 101[rand], 5M: 105[rand]
            rnd_suffix = random.randint(10, 99)
            lvl_id = 1 if counts == 1000000 else 5
            seed = int(f"{id_counter}{lvl_id}{rnd_suffix}")
            
            commands.append(create_job(out_path, counts, seed))
        
        id_counter += 1

# Escribir el script generator
script_name = "generate_pilot_commands.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Script generado para crear dataset piloto\n")
    f.write("# Ejecutar linea por linea o con un job array\n\n")
    f.write("\n".join(commands))
    f.write("\n")

print(f"✅ Archivo '{script_name}' generado con {len(commands)} tareas de simulación.")
print(f"   - Target Full: 1")
print(f"   - Train Pairs: {N_TRAIN} x {len(INPUT_LEVELS)}")
print(f"   - Val Pairs:   {N_VAL} x {len(INPUT_LEVELS)}")
print(f"   - Total Sims:  {1 + (N_TRAIN + N_VAL) * len(INPUT_LEVELS)}")
