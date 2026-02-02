# Proyecto Modular 3 - Guía de Inicio Rápido

## ¿Qué es este proyecto?

Sistema completo de simulación Monte Carlo de aceleradores lineales (linacs) de 6 MV para radioterapia, con denoising de dosis mediante redes neuronales profundas (MCDNet 3D CNN).

## Instalación Rápida

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd Modular3

# 2. Crear entorno conda
conda env create -f environment.yml
conda activate modular3

# 3. O instalar con pip
pip install -r requirements.txt
```

## Uso Básico

### Opción 1: Workflow Completo Automatizado

```bash
# Ejecutar todo el pipeline de una vez
bash scripts/run_complete_workflow.sh
```

### Opción 2: Paso a Paso

#### 1. Generar Phase Space del Linac

```bash
python simulations/linac_6mv.py \
    --energy 5.8 \
    --particles 1e8 \
    --output data/phase_space/linac_ps.root
```

#### 2. Generar Dataset de Entrenamiento

```bash
python data/dataset_generator.py \
    --output data/training/ \
    --phantoms water bone lung \
    --fields 5 10 15 \
    --samples-per-config 20
```

#### 3. Entrenar MCDNet

```bash
python models/training.py \
    --data-dir data/training/ \
    --epochs 100 \
    --batch-size 4 \
    --device cuda
```

#### 4. Aplicar Denoising

```bash
python models/inference.py \
    --model models/checkpoints/mcdnet_best.pth \
    --input data/noisy_dose.mhd \
    --output results/denoised_dose.mhd
```

#### 5. Validar con Gamma Index

```bash
python analysis/gamma_index.py \
    --reference data/reference_dose.mhd \
    --evaluated results/denoised_dose.mhd \
    --criteria 3%/3mm \
    --output results/gamma/
```

### Opción 3: Ejemplo Interactivo

```bash
python examples/complete_workflow.py
```

## Estructura del Proyecto

```
Modular3/
├── simulations/          # Módulos GATE 10
│   ├── linac_6mv.py           # Simulación del linac
│   ├── phase_space.py         # Generación de phase space
│   └── dose_calculation.py    # Cálculo de dosis en fantomas
│
├── models/               # Arquitecturas de IA
│   ├── mcdnet.py             # Red MCDNet 3D
│   ├── training.py           # Pipeline de entrenamiento
│   └── inference.py          # Inferencia y exportación ONNX
│
├── analysis/             # Análisis y validación
│   ├── gamma_index.py        # Análisis gamma
│   ├── visualization.py      # Gráficos (PDD, perfiles)
│   └── metrics.py            # Métricas (MSE, PSNR, SSIM)
│
├── config/               # Configuración YAML
│   ├── linac_params.yaml     # Parámetros del linac
│   ├── physics.yaml          # Física Geant4
│   └── materials.yaml        # Conversión HU→material
│
├── data/                 # Datos
│   ├── phantoms/             # Geometrías de fantomas
│   ├── phase_space/          # Archivos .root
│   └── training/             # Dataset de entrenamiento
│
├── docs/                 # Documentación
│   ├── latex/main.tex        # Reporte académico
│   └── METODOLOGIA.md        # Guía de calibración
│
└── tests/                # Tests unitarios
    └── test_simulations.py
```

## Comandos Make

```bash
make help           # Ver todos los comandos
make install        # Instalar dependencias
make run-tests      # Ejecutar tests
make train          # Entrenar modelo
make validate       # Validar con gamma
make docs           # Compilar LaTeX
make clean          # Limpiar temporales
make workflow       # Ejecutar workflow completo
```

## Requisitos del Sistema

- **Python:** 3.9-3.12
- **GPU:** CUDA-compatible (recomendado para entrenamiento)
- **RAM:** Mínimo 16 GB (32 GB recomendado)
- **Disco:** ~50 GB para datasets completos

## Dependencias Principales

- **GATE 10** (OpenGate): Simulación Monte Carlo
- **PyTorch 2.0+**: Deep learning
- **SimpleITK**: Procesamiento de imágenes médicas
- **PyMedPhys**: Análisis gamma index
- **NumPy, Matplotlib**: Análisis y visualización

## Validación

El proyecto cumple con criterios clínicos:

- ✅ **Gamma Index 3%/3mm:** Pass rate ≥ 95%
- ✅ **Gamma Index 2%/2mm:** Pass rate ≥ 90%
- ✅ **PSNR:** > 40 dB
- ✅ **Diferencia de dosis:** < 1% (media)

## Fechas Importantes (CUCEI)

- **27 de marzo 2026:** Entrega de código
- **5-9 de mayo 2026:** Presentación oral (20 min)

## Soporte

Para problemas o preguntas:
1. Revisar documentación en `docs/`
2. Ejecutar tests: `make run-tests`
3. Ver ejemplos en `examples/`

## Licencia

MIT License - Ver `LICENSE` para detalles

---

**Proyecto Modular 3**  
CUCEI - Universidad de Guadalajara  
Licenciatura en Física  
Enero 2026
