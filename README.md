# Modular3 - Simulación de Dosis Monte Carlo

Simulación de dosis en agua usando OpenGate/Geant4 con phase-space IAEA.

## Estructura

```
Modular3/
├── data/IAEA/                    # Phase-space files
│   └── Varian_Clinac_2100CD_6MeV_15x15.root
├── simulations/
│   └── dose_simulation.py        # Script principal
├── output/                       # Resultados
├── launch_simulations.sh         # Launcher cluster SLURM
└── run_local.sh                  # Test local
```

## Uso Local

```bash
# Activar entorno
source .venv/bin/activate

# Test rápido (dry-run)
./run_local.sh --dry-run

# Simulación local
./run_local.sh 10000 4    # 10k partículas, 4 threads
```

## Uso Cluster (SLURM)

```bash
# 100 simulaciones, 1M partículas, 2 concurrent
./launch_simulations.sh 1-100 1000000 2

# 25 simulaciones, 10M partículas, 4 concurrent
./launch_simulations.sh 1-25 10000000 4
```

## Resultados

Cada simulación genera en `output/pair_XXX/`:
- `dose.mhd` + `dose.raw` - Mapa de dosis 3D
- `info.json` - Metadatos

## Configuración

- Fantoma: agua 30×30×30 cm
- Malla: 2mm XY, 1mm Z (150×150×300 voxels)
- Física: QGSP_BIC_EMZ
