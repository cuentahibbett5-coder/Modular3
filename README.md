# Proyecto Modular 3: Simulaciones de Radioterapia con GATE 10 e IA

## Descripción del Proyecto

Implementación de simulaciones Monte Carlo de alta fidelidad para radioterapia de 6 MV utilizando GATE 10 (OpenGate) con integración de modelos de Inteligencia Artificial para denoising de distribuciones de dosis. Este proyecto forma parte del Proyecto Modular 3 del Departamento de Física de CUCEI.

## Características Principales

- **Simulación Monte Carlo**: Modelado completo de acelerador lineal (Linac) de 6 MV con Geant4 11.3.2
- **Gestión de Phase Space**: Generación y reutilización de espacios de fase para optimización computacional
- **Phantoms Voxelizados**: Integración de imágenes CT con calibración estequiométrica de Unidades Hounsfield
- **Modelos de IA**: Arquitectura MCDNet 3D para denoising de dosis Monte Carlo
- **Validación Clínica**: Análisis mediante índice Gamma (3%/3mm, 2%/2mm)

## Estructura del Proyecto

```
Modular3/
├── simulations/          # Módulos de simulación GATE 10
│   ├── linac_6mv.py     # Modelado del acelerador lineal
│   ├── phase_space.py   # Generación de espacios de fase
│   ├── dose_calculation.py  # Cálculo de dosis con DoseActor
│   └── geometry/        # Definiciones de geometría
├── models/              # Modelos de Inteligencia Artificial
│   ├── mcdnet.py       # Arquitectura MCDNet para denoising
│   ├── training.py     # Pipeline de entrenamiento
│   ├── inference.py    # Inferencia y exportación ONNX
│   └── utils.py        # Utilidades para modelos
├── data/               # Datos de simulación
│   ├── phantoms/       # Phantoms CT y agua
│   ├── phase_space/    # Archivos de espacio de fase
│   └── dose_maps/      # Mapas de dosis generados
├── analysis/           # Análisis y validación
│   ├── gamma_index.py  # Cálculo de índice Gamma
│   ├── visualization.py # Visualización de dosis
│   └── metrics.py      # Métricas de evaluación
├── config/             # Archivos de configuración
│   ├── linac_params.yaml    # Parámetros del acelerador
│   ├── physics.yaml         # Configuración de física
│   └── materials.yaml       # Tabla de materiales HU
├── docs/               # Documentación
│   ├── latex/          # Plantilla LaTeX para reporte
│   └── metodologia.md  # Metodología detallada
└── tests/              # Pruebas de validación
    └── test_simulations.py
```

## Requisitos del Sistema

### Software Requerido

- Python 3.9 - 3.12
- GATE 10 (OpenGate)
- Geant4 11.3.2 (incluido con OpenGate)
- CUDA 11.8+ (opcional, para entrenamiento de IA)

### Dependencias Python

Ver `requirements.txt` para lista completa. Principales:

- `opengate` >= 10.0
- `numpy` >= 1.24
- `torch` >= 2.0 (con soporte CUDA)
- `SimpleITK` >= 2.2
- `pymedphys` >= 0.39
- `matplotlib` >= 3.7
- `onnx` >= 1.14

## Instalación

### 1. Crear Entorno Virtual

```bash
python3 -m venv venv_modular3
source venv_modular3/bin/activate  # Linux/Mac
# o
venv_modular3\Scripts\activate     # Windows
```

### 2. Instalar Dependencias

```bash
# Instalación básica (sin visualización para clusters)
pip install "opengate[novis]"

# O instalación completa con visualización
pip install opengate

# Instalar el resto de dependencias
pip install -r requirements.txt
```

### 3. Descargar Datos de Geant4

```bash
# En caso de problemas SSL
export GIT_SSL_NO_VERIFY=1

# Verificar instalación y descargar bases de datos
opengate_info
```

### 4. Ejecutar Pruebas de Validación

```bash
# Ejecutar suite de pruebas de OpenGate
opengate_tests

# Ejecutar pruebas del proyecto
pytest tests/
```

## Uso Básico

### Simular Acelerador Linac de 6 MV

```python
from simulations.linac_6mv import LinacSimulation

# Configurar simulación
sim = LinacSimulation(energy_MeV=5.8, spot_size_mm=3.0)
sim.setup_geometry()
sim.setup_physics()
sim.run(num_particles=1e8)
```

### Generar Espacio de Fase

```python
from simulations.phase_space import PhaseSpaceGenerator

psg = PhaseSpaceGenerator()
psg.generate(linac_config='config/linac_params.yaml', 
             output='data/phase_space/linac_6mv.root')
```

### Convertir IAEA PHSP a ROOT

```bash
# Convertir archivo IAEA experimental a formato ROOT compatible con OpenGate
python simulations/convert_iaea_to_root.py \
    --input data/iaea_phsp/Varian_6MeV.IAEAphsp \
    --output data/phase_space/varian_6mv.root \
    --max-particles 5000000

# Variables generadas: Ekine, X, Y, Z, dX, dY, dZ, Weight, ParticleType
```

### Usar Phase Space Experimental en Simulación

```python
import opengate as gate

sim = gate.Simulation()

# Usar phase space IAEA convertido como fuente
source = sim.add_source('PhaseSpaceSource', 'phsp_source')
source.phsp_file = 'data/phase_space/varian_6mv.root'
source.particle = ''  # Auto-detectado del archivo ROOT
source.position.translation = [0, 0, 0]  # cm (ajustar según geometría)

# Continuar con phantom y actors...
```

### Calcular Dosis en Phantom

```python
from simulations.dose_calculation import DoseCalculator

calc = DoseCalculator(
    phantom_path='data/phantoms/water_phantom.mhd',
    phase_space='data/phase_space/linac_6mv.root'
)
dose_map = calc.calculate_dose(num_particles=1e9)
calc.save_dose('data/dose_maps/dose_result.mhd')
```

### Entrenar Modelo de IA

```python
from models.training import MCDNetTrainer

trainer = MCDNetTrainer(
    data_dir='data/dose_maps',
    model_save_path='models/checkpoints'
)
trainer.train(epochs=100, batch_size=4)
```

### Aplicar Denoising

```python
from models.inference import DoseDenoiser

denoiser = DoseDenoiser(model_path='models/checkpoints/mcdnet_best.pth')
clean_dose = denoiser.denoise(noisy_dose_array)
```

### Validar con Índice Gamma

```python
from analysis.gamma_index import GammaAnalysis

gamma = GammaAnalysis(
    reference='data/dose_maps/reference.mhd',
    evaluated='data/dose_maps/predicted.mhd'
)
pass_rate = gamma.calculate(dose_diff_percent=3, dta_mm=3)
print(f"Pass rate: {pass_rate:.2f}%")
```

## Parámetros Óptimos del Linac

### Haz de Electrones Primario

| Parámetro | Valor Recomendado | Efecto Principal |
|-----------|-------------------|------------------|
| Energía Media | 5.8 MeV | Profundidad del máximo de dosis |
| FWHM Energía | 3% | Modulación del espectro |
| Spot Size (FWHM) | 3 mm | Penumbra y perfiles laterales |
| Distribución Espacial | Gaussiana | Dispersión natural del haz |

### Configuración de Física Geant4

- **Physics List**: `QGSP_BIC_EMZ` o `emstandard_opt3`
- **Cortes de Producción (World)**: 1.0 mm
- **Cortes de Producción (Phantom)**: 0.1 - 1.0 mm
- **Límite de Paso**: 0.5 - 1.0 mm

## Validación y Métricas

### Criterios de Aceptación

- **Índice Gamma 3%/3mm**: Pass rate > 95%
- **Índice Gamma 2%/2mm**: Pass rate > 90%
- **Error RMS en región de alto gradiente**: < 2%
- **Tiempo de inferencia IA**: < 500 ms para volumen 3D

## Requisitos Académicos (CUCEI)

### Proyecto Modular 3

- **Fecha límite de envío**: 27 de marzo, 11:00 am
- **Presentación pública**: 5-9 de mayo
- **Formato póster**: 90 x 120 cm (vertical)
- **Evaluación**: Acreditado/No Acreditado

### Documentación Requerida

1. Documento técnico en LaTeX (plantilla en `docs/latex/`)
2. Póster científico para presentación
3. Código fuente documentado
4. Resultados de validación con índice Gamma

## Contribución y Desarrollo

### Ejecutar en Modo Debug

```bash
# Simular con menos partículas para pruebas rápidas
python simulations/linac_6mv.py --debug --particles 1e6
```

### Generación de Datos para IA

```bash
# Generar dataset de entrenamiento
python data/dataset_generator.py --num-samples 1000 --output data/training_set
```

## Referencias

### GATE 10 y OpenGate

- Documentación oficial: https://opengate-python.readthedocs.io/
- GitHub: https://github.com/OpenGATE/opengate
- Geant4 Physics Reference: https://geant4-userdoc.web.cern.ch/

### Publicaciones Relevantes

- Sarrut et al. (2024). "GATE 10: Modeling radiation therapy with Geant4 and Python"
- MCDNet: "Deep convolutional neural network for denoising Monte Carlo dose distributions"
- Gamma Index: Low et al. (1998). "A technique for the quantitative evaluation of dose distributions"

## Licencia

Este proyecto es desarrollado como parte de los requisitos académicos del Departamento de Física de CUCEI, Universidad de Guadalajara.

## Contacto y Soporte

Para consultas relacionadas con el proyecto:
- Asesor: [Nombre del asesor]
- Institución: CUCEI - Universidad de Guadalajara
- Departamento: Física

## Notas Importantes

⚠️ **Advertencias**:
- Las simulaciones Monte Carlo requieren recursos computacionales significativos
- Se recomienda usar clusters HPC para generación de datasets masivos
- Los archivos de phase space pueden ocupar varios GB de almacenamiento
- Verificar instalación de CUDA para entrenamiento de modelos de IA

📊 **Estadísticas del Proyecto**:
- Tiempo estimado de simulación: 2-10 horas por configuración
- Tiempo de entrenamiento IA: 12-24 horas con GPU
- Tiempo de inferencia: ~200-500 ms por volumen 3D
- Espacio en disco requerido: ~50-100 GB para datasets completos
