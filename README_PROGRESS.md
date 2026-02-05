# Modular3 - Simulación Dosis MCDNet

## 🎯 Objetivos
Generar mapas de dosis 3D para entrenar la red neuronal MCDNet usando simulaciones OpenGate con la fuente de fase IAEA completa del Clinac Varian 2100CD 6MeV.

## 📊 Dataset Actual
- **Partículas**: 29,288,306 (29.3M después de filtrado)
  - Fotones: 9,266,454 (31.6%)
  - Electrones: 20,020,804 (68.3%)
  - Positrones: 1,048 (0.004%)
- **Espectro**: 6 MeV nominal (Varian Clinac 2100CD)
- **Geometría**: Clinac 15×15 campo

## 🏗️ Estructura del Proyecto

```
Modular3/
├── data/
│   ├── IAEA/
│   │   ├── Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp       # 1.01 GB - Binario IAEA
│   │   ├── Varian_Clinac_2100CD_6MeV_15x15.IAEAheader    # Header IAEA
│   │   └── Varian_Clinac_2100CD_6MeV_15x15_FULL.root     # 659 MB - Convertido ROOT
│   └── ...
├── simulations/
│   └── dose_simulation.py                                 # Script simulación principal
├── iaea_to_root.C                                        # Conversor IAEA↔ROOT (C++)
├── run_test_simulation.sh                                # Test con 10k partículas
├── run_full_simulation.sh                                # Simulación completa
├── output/
│   ├── dose_maps/
│   │   ├── test_10k/                                     # ✅ Completada (6.7s)
│   │   │   ├── dose_edep.raw
│   │   │   ├── dose_edep.mh
│   │   │   └── info.json
│   │   └── full_1M/                                      # 🔄 En progreso...
│   └── test_10k.log
└── ...
```

## 🔧 Flujo de Trabajo

### 1️⃣ Conversión IAEA → ROOT
```bash
# Convertir archivo IAEA binario a formato ROOT
cd /home/fer/fer/Modular3
root -l -b -q iaea_to_root.C
# Parámetros:
# - Input: data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp
# - Output: data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root
# - Filtro: Solo PIDs válidos (11, -11, 22)
```

**Estructura del registro IAEA (37 bytes):**
- 1 byte: Tipo de partícula (1=fotón, 2=electrón, 3=positrón, 253-255=descartados)
- 6 floats (24 bytes): X, Y, Z, U, V, Weight
- 2 ints (8 bytes): History number, ILB PENELOPE variable
- 4 bytes: Padding

**Mapeo de tipos:**
```
type byte=1 → PDG=22 (Fotones)
type byte=2 → PDG=11 (Electrones)  
type byte=3 → PDG=-11 (Positrones)
type byte∈{253,254,255} → Filtrados (PID=0)
```

### 2️⃣ Simulación de Dosis
```bash
# Test rápido: 10k partículas
/home/fer/fer/Modular3/.venv/bin/python simulations/dose_simulation.py \
    --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root \
    --output output/dose_maps/test_10k \
    --n-particles 10000 \
    --threads 4 \
    --seed 42

# Simulación completa: 29.3M partículas
/home/fer/fer/Modular3/.venv/bin/python simulations/dose_simulation.py \
    --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root \
    --output output/dose_maps/full_all \
    --n-particles 29288306 \
    --threads 8 \
    --seed 42

# Con opciones personalizadas
/home/fer/fer/Modular3/.venv/bin/python simulations/dose_simulation.py \
    --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root \
    --output output/dose_maps/custom \
    --n-particles 1000000 \
    --threads 8 \
    --seed 42 \
    --spacing-xy 1.0 \          # Resolución XY en mm
    --spacing-z 0.5 \           # Resolución Z en mm
    --gap 50                    # Gap aire-agua en mm
```

**Parámetros de simulación:**
- `--input`: Archivo PHSP ROOT
- `--output`: Directorio de salida
- `--n-particles`: Número de partículas a simular
- `--threads`: Hilos paralelos (1-8 recomendado)
- `--seed`: Seed pseudoaleatorio
- `--spacing-xy`: Resolución grid XY (default 2.0mm)
- `--spacing-z`: Resolución grid Z (default 1.0mm)
- `--gap`: Distancia aire-agua (default 50mm)
- `--dry-run`: Solo mostrar configuración sin ejecutar

### 3️⃣ Geometría OpenGate
- **Mundo**: 400×400×600mm (Aire)
- **Fuente PHSP**: Z ≈ 78.5mm
- **Gap aire**: 50mm (ajustable)
- **Fantoma agua**: 300×300×300mm
  - Superficie: Z = 28.5mm
  - Centro: Z = -121.5mm
- **Grid dosis**: 150×150×300 voxeles (51.5MB @ 2×2×1mm)

## ✅ Estado Actual

| Hito | Estado | Detalles |
|------|--------|----------|
| Archivo IAEA | ✅ Verificado | 29.4M partículas, 1.01 GB |
| Conversión a ROOT | ✅ Completada | 659 MB, 29.3M partículas válidas |
| Test 10k partículas | ✅ Exitosa | 6.7s, mapas generados |
| Simulación 1M | 🔄 En progreso | ~1-2 horas ETA |
| Simulación 29.3M | ⏳ Pendiente | ~30-40 horas (todo el dataset) |

## 📈 Pasos Siguientes

1. **Completar simulación 1M** → Verificar estadísticas y validación física
2. **Generar múltiples datasets** → Diferentes seeds para variabilidad
3. **Calcular métricas de dosis** → DVH, uniformidad, etc.
4. **Preparar datos para MCDNet** → Normalización, formato TensorFlow
5. **Entrenar MCDNet** → Validación cross-validation

## 📝 Notas Técnicas

### Decodificación de Tipos Corregida
Inicialmente se asumía que el byte `type` contenía tipo=1→e-, type=2→e+, type=3→γ, pero el análisis de datos mostró:
- **Versión anterior** (incorrecta): ~1k fotones en archivo
- **Verificación contra header** IAEA: 9.4M fotones esperados
- **Solución**: Invertir mapeo → type=1→γ, type=2→e-, type=3→e+
- **Resultado**: Distribución correcta ✅ 9.2M γ, 20M e-, 1k e+

### Filtrado de Partículas Descartadas
- ~118k partículas (0.4%) con types desconocidas (253, 254, 255)
- OpenGate requiere PIDs válidos → Filtered from ROOT
- 29.4M → 29.3M partículas guardadas

### Estructura ROOT Creada
```
Tree "phsp" con branches:
├── pid (I): PDG code (11, -11, 22)
├── E (F): Energía
├── x, y, z (F): Posición [mm]
├── dx, dy, dz (F): Cosenos directores
├── w (F): Peso estadístico
├── history (I): Número de historia
├── ilb (I): Variable ILB PENELOPE
└── newHist (O): Flag nueva historia
```

## 🖥️ Dependencias Instaladas
- OpenGate 10.0.3
- ROOT 6.x
- Python 3.12.3 (venv)
- numpy, scipy, matplotlib
- uproot (lectura ROOT desde Python)

## 💾 Archivos Importantes

| Archivo | Tamaño | Descripción |
|---------|--------|-------------|
| `Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp` | 1.01 GB | Datos binarios IAEA originales |
| `Varian_Clinac_2100CD_6MeV_15x15_FULL.root` | 659 MB | Convertido a ROOT, 29.3M partículas |
| `iaea_to_root.C` | ~170 KB | Script conversor C++ |
| `simulations/dose_simulation.py` | ~6 KB | Motor simulación principal |

---
**Actualizado**: Feb 4, 2025
**Próximas acciones**: Monitorear simulación 1M y documentar resultados

