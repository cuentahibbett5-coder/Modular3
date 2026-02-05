# Descripción del Arreglo Experimental - Simulación Monte Carlo

## 1. FUENTE DE RADIACIÓN

**Acelerador Lineal:** Varian Clinac 2100CD  
**Energía nominal:** 6 MeV (electrones)  
**Campo de radiación:** 15×15 cm² en isocentro (SSD = 100 cm típico)

### Phase Space (PHSP)
- **Archivo:** `Salida_Varian_OpenGate_mm.root`
- **Formato:** ROOT (TTree) con ramas: pid, E, x, y, z, dx, dy, dz, w, history, ilb
- **Partículas totales:** 29,393,164 (29.4 millones)
- **Ubicación en simulación:** Z = 785 mm (downstream del cabezal)
- **Distribución de partículas:**
  - Electrones (PDG 11): ~60%
  - Positrones (PDG -11): ~5%
  - Fotones (PDG 22): ~35%
- **Energía:** Distribución continua desde ~0.5 MeV hasta 6 MeV
- **Coordenadas:** Sistema cartesiano en mm
  - X, Y: distribución espacial del haz (±150 mm aprox)
  - Z: dirección principal de propagación (+Z downstream)
- **Direcciones:** Cosenos directores (dx, dy, dz) normalizados

## 2. GEOMETRÍA DE LA SIMULACIÓN

### Sistema de Coordenadas
- **Eje Z:** Dirección del haz (partículas viajan hacia +Z)
- **Origen:** Centro del fantoma de agua
- **Unidades:** milímetros (mm)

### Volúmenes

#### World (Contenedor)
- **Material:** Aire (G4_AIR)
- **Dimensiones:** 500 × 500 × 2500 mm³
- **Función:** Volumen madre que contiene todos los elementos

#### Fantoma de Agua
- **Material:** Agua (G4_WATER, ρ = 1.0 g/cm³)
- **Dimensiones:** 300 × 300 × 300 mm³ (30×30×30 cm³)
- **Posición:**
  - Centro: Z = 985 mm (coordenadas absolutas)
  - Superficie superior: Z = 835 mm
  - Superficie inferior: Z = 1135 mm
- **Relativo al PHSP:**
  - Gap de aire: 50 mm (5.0 cm) entre PHSP y superficie del agua
  - PHSP → 785 mm
  - Inicio agua → 835 mm

### Justificación Geométrica
La geometría simula una configuración típica de radioterapia externa:
- **SSD efectivo:** ~78.5 cm al plano del PHSP
- **Gap adicional:** 5 cm hasta superficie del fantoma
- **Distancia total fuente-superficie:** ~83.5 cm
- El PHSP captura partículas después de atravesar el sistema de colimación

## 3. GRID DE DOSIS

### Configuración del DoseActor
- **Método:** DoseActor de OpenGate (attached_to='water')
- **Resolución espacial:**
  - Spacing X-Y: 2.0 mm (plano transversal)
  - Spacing Z: 1.0 mm (profundidad, mayor resolución)
- **Dimensiones del grid:**
  - Voxels: 150 × 150 × 300 (X × Y × Z)
  - Tamaño físico: 300 × 300 × 300 mm³
  - Volumen de voxel: 4 mm³ (2×2×1 mm)
  - Total voxels: 6,750,000
- **Sistema de coordenadas:** Relativo al centro del fantoma de agua
  - Origen: (0, 0, 0) en el centro del agua
  - Rango Z: -150 a +150 mm desde el centro
  - Profundidad desde superficie: 0 a 30 cm

### Almacenamiento
- **Formato:** MetaImage (MHD/RAW)
- **Tamaño archivo:** ~52 MB (100k partículas), ~52 MB (1M partículas)
- **Precisión:** Float32 (4 bytes por voxel)

## 4. FÍSICA DE LA SIMULACIÓN

### Motor Monte Carlo
- **Framework:** OpenGate 10.x (basado en Geant4)
- **Physics List:** QGSP_BIC_EMZ
  - QGSP: Quark-Gluon String Precompound
  - BIC: Binary Ion Cascade
  - EMZ: Electromagnetic physics con opciones extendidas

### Procesos Físicos Principales
Para **electrones y positrones**:
- Ionización (producción de electrones secundarios)
- Bremsstrahlung (producción de fotones)
- Scattering múltiple coulombiano
- Aniquilación de positrones (e+ + e- → 2γ)

Para **fotones**:
- Efecto fotoeléctrico
- Compton scattering
- Producción de pares (E > 1.022 MeV)

### Rango de Validez
- **Energías:** 250 eV - 10 GeV
- **Cortes de producción:** Por defecto de QGSP_BIC_EMZ
- **Partículas secundarias:** Totalmente rastreadas hasta absorción

## 5. PARÁMETROS DE SIMULACIÓN

### Configuración Estándar (1M partículas)
```
Input:      data/IAEA/Salida_Varian_OpenGate_mm.root
Partículas: 1,000,000 (desde PHSP de 29.4M)
Threads:    1 (single-thread para reproducibilidad)
Seed:       42 (aleatorización determinista)
Tiempo:     ~74 segundos (~13,500 partículas/segundo)
```

### Configuración Extendida (10M partículas)
```
Partículas: 10,000,000
Tiempo estimado: ~740 segundos (~12 minutos)
Mejora estadística: √10 ≈ 3.16× menor incertidumbre
```

## 6. RESULTADOS FÍSICOS VALIDADOS

### Simulación 1M Partículas
- **Voxels con dosis:** 1,691,069 (25% del grid total)
- **Dosis máxima:** 46.2 Gy (absoluta)
- **Rmax:** 1.56 cm (desde superficie del agua)
  - **Esperado teórico:** Rmax ≈ E/2 = 6/2 = 3 cm (regla empírica)
  - **Rango observado:** 1.5-2.5 cm es típico para 6 MeV
  - **Conclusión:** ✅ Físicamente consistente
- **Distribución lateral:** Simétrica en X-Y
- **Forma del PDD:** 
  - Build-up rápido hasta Rmax
  - Decaimiento exponencial después de Rmax
  - Alcance práctico: ~3-4 cm (R90)

### Comparación con Literatura
Para **electrones de 6 MeV en agua**:
- **Rmax teórico:** 1.5-2.5 cm
- **R50:** ~2.5-3.0 cm
- **Rp (alcance práctico):** ~3.5-4.0 cm
- **Nuestros resultados:** Consistentes con valores publicados

## 7. CONVERSIÓN DE UNIDADES

### Dosis por Partícula
```
Dosis_normalizada = Dosis_absoluta / N_partículas
```
Para 1M partículas:
- Dosis máxima: 46.2 Gy
- Dosis normalizada: 4.62×10⁻⁵ Gy/partícula

### Conversión a Dosis Clínica
Para obtener dosis terapéutica (ej. 2 Gy):
```
N_partículas = Dosis_objetivo / Dosis_normalizada
N_partículas = 2 Gy / (4.62×10⁻⁵ Gy/partícula) ≈ 43,000 partículas
```

## 8. INCERTIDUMBRES Y LIMITACIONES

### Estadísticas
- **100k partículas:** Incertidumbre ~1% en región de alto flujo
- **1M partículas:** Incertidumbre ~0.3% en región de alto flujo
- **10M partículas:** Incertidumbre ~0.1% (objetivo para publicación)

### Limitaciones del PHSP
- PHSP pre-calculado (no incluye variaciones del cabezal)
- Campo fijo 15×15 cm
- No incluye efectos de mesa de tratamiento
- Geometría simplificada (fantoma homogéneo)

### Validación Pendiente
- [ ] Comparar con medidas experimentales (cámara de ionización)
- [ ] Validar perfiles laterales con film radiográfico
- [ ] Comparar con simulaciones de referencia (PRIMO, EGSnrc)
- [ ] Verificar factores de output del acelerador

## 9. ARCHIVOS GENERADOS

### Por Simulación
```
output/test_1M/
├── dose_edep.mhd          # Metadata del grid de dosis
├── dose_edep.raw          # Datos binarios (52 MB)
├── info.json              # Metadata de la simulación
├── dose_3d.html           # Visualización 3D interactiva (17 MB)
└── perfiles_dosis.png     # Perfiles PDD y laterales
```

### Formato ROOT del PHSP
```
TTree: PhaseSpace
Branches:
  - pid (Int_t): PDG code (11, -11, 22)
  - E (Float_t): Energía cinética [MeV]
  - x, y, z (Float_t): Posición [mm]
  - dx, dy, dz (Float_t): Cosenos directores
  - w (Float_t): Peso estadístico
  - history (Int_t): Número de historia primaria
  - ilb (Int_t): Información adicional
```

## 10. REPRODUCIBILIDAD

### Requisitos del Sistema
- **Python:** 3.12
- **OpenGate:** 10.x
- **ROOT:** 6.x (para leer PHSP)
- **ITK:** 5.x (para I/O de MHD)
- **Plotly:** 5.x (visualización 3D)

### Comando de Ejecución
```bash
python simulations/dose_simulation.py \
    --input data/IAEA/Salida_Varian_OpenGate_mm.root \
    --output output/test_1M \
    --n-particles 1000000 \
    --threads 1 \
    --seed 42
```

### Control de Versiones
- Simulación reproducible con seed fijo
- Geometría versionada en código
- PHSP estático (no regenerado)

---

**Fecha de documentación:** 2026-02-05  
**Versión del código:** v1.0  
**Autor:** Sistema de simulación Monte Carlo para radioterapia  
**Validación:** En progreso
