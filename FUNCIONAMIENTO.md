# Funcionamiento Preciso del Experimento Modular3

## 🎯 Objetivo del Experimento

Simular un acelerador lineal médico (Elekta Versa HD) para obtener las características del haz de fotones en el plano de tratamiento (SSD = 100 cm).

---

## ⚙️ Proceso de la Simulación

### 1. Generación del Haz Primario
```
Fuente de electrones: 6 MeV
↓
Posición: (0, 0, 0) - origen en el target
Distribución: Gaussiana (σ = 1 mm)
```

### 2. Interacción en el Target (Tungsteno)
```
Electrones (6 MeV) + Target (W)
↓
Bremsstrahlung → Fotones de rayos X
Espectro: 0 - 6 MeV (continuo)
Eficiencia: ~1-2% de conversión
```

### 3. Modificación del Haz

**a) Filtro Aplanador**
- Material: Cobre/Acero
- Función: Homogenizar el perfil del haz
- Efecto: Haz más uniforme lateralmente, pero reduce intensidad

**b) Colimador Primario**
- Material: Tungsteno/Plomo
- Función: Conformar haz inicial
- Reduce dispersión fuera del campo útil

**c) Cámara de Ionización**
- Función: Monitorear dosis en tiempo real
- No modifica significativamente el haz

**d) Jaws (Colimadores X/Y)**
- Material: Tungsteno
- Función: Definir campo rectangular
- Campo actual: 10×10 cm² en isocentro

**e) MLC (Multi-Leaf Collimator)**
- 80 láminas de tungsteno
- Función: Conformar campo a forma de tumor
- Precisión: ±1 mm

### 4. Captura en Phase Space
```
Plano a z = -1000 mm (100 cm del target)
↓
Por cada partícula que cruza:
  - Posición (x, y, z)
  - Dirección (dx, dy, dz)
  - Energía cinética
  - Tipo de partícula (fotón/electrón)
  - Peso estadístico
```

---

## 📊 Resultados de la Simulación

**Archivo generado:** `data/phase_space/versa_6mv_1e6.root`

**Contenido:**
- **Partículas totales:** 36,276 (de 1,000,000 simuladas)
- **Composición:**
  - 99.4% fotones
  - 0.6% electrones (dispersión Compton)
- **Energía promedio:** ~2 MeV
- **Distribución espacial:** Gaussiana, FWHM ~5 cm

---

## 🔍 Lo que NO muestra la visualización actual

**PyVista muestra:**
- ✅ Geometría del linac
- ✅ Posiciones finales en phase space
- ❌ **Trayectorias reales** dentro del linac

**¿Por qué?**
- El `PhaseSpaceActor` solo guarda el **estado final** de las partículas
- Las trayectorias intermedias se pierden (para ahorrar espacio)
- Una partícula puede hacer cientos de dispersiones antes de llegar al phase space

---

## 🎬 Para ver trayectorias reales

Necesitarías agregar un `TrackingActor` que guarde cada paso de cada partícula:

```python
# En linac_6mv.py
tracking = sim.add_actor('TrackingActor', 'tracking')
tracking.attached_to = 'world'
tracking.output_filename = 'output/tracks.root'
tracking.track_types_flag = True

# ADVERTENCIA: Genera archivos MUY grandes
# 1M partículas × 100 pasos/partícula = 100M registros
# Tamaño estimado: ~10-50 GB
```

Luego podrías visualizar las trayectorias reales, pero solo es viable para **pocas partículas** (~1000).

---

## 📈 Aplicaciones Prácticas

Con el phase space generado puedes:

### 1. Calcular Dosis en Fantoma
```python
# Agregar fantoma de agua
phantom = sim.add_volume('Box', 'phantom')
phantom.material = 'G4_WATER'
phantom.size = [30, 30, 30]  # cm

# Agregar actor de dosis
dose = sim.add_actor('DoseActor', 'dose')
dose.attached_to = 'phantom'
dose.output_filename = 'output/dose.mhd'
dose.size = [300, 300, 300]  # voxels
```

### 2. Analizar Distribución Espacial
```python
# Ya lo hiciste con quick_view.py
# Histogramas, perfiles, fluencia
```

### 3. Comparar con Mediciones
```python
# Comparar con datos de comisionamiento
# PDD (Percent Depth Dose)
# Perfiles laterales
```

### 4. Optimizar Tratamientos
```python
# Usar phase space como fuente
# Simular diferentes geometrías de paciente
# Calcular planes IMRT
```

---

## 🚀 Próximos Pasos Sugeridos

### Opción 1: Cálculo de Dosis
Ver cómo el haz deposita energía en un fantoma de agua (PDD, perfiles)

### Opción 2: Variar Parámetros
- Cambiar tamaño de campo (5×5, 20×20 cm²)
- Cambiar energía (10 MV, 15 MV)
- Simular IMRT (campos modulados)

### Opción 3: Validación
Comparar con datos experimentales del linac real

### Opción 4: Visualización Detallada
Simular pocas partículas (1000) con TrackingActor para ver trayectorias reales

---

## 📚 Física del Proceso

**Interacciones principales:**

1. **Bremsstrahlung** (electrones → fotones)
   - En el target de W
   - Eficiencia proporcional a Z (Z_W = 74)

2. **Efecto Compton** (fotones → electrones)
   - Dispersión inelástica
   - Genera electrones secundarios

3. **Efecto Fotoeléctrico**
   - Dominante a bajas energías (<100 keV)
   - Fotón absorbido completamente

4. **Producción de Pares** (fotones → e⁺e⁻)
   - Solo si E_γ > 1.022 MeV
   - Relevante en materiales pesados (W, Pb)

---

## 🔧 Herramientas del Proyecto

```bash
# Simulación principal
python simulations/linac_6mv.py

# Visualización geometría
python simulations/visualize_pyvista.py

# Visualización con haz
python simulations/visualize_pyvista.py --beam --n-particles 5000

# Análisis rápido
python simulations/quick_view.py

# Visualización phase space
python simulations/visualize_pyvista.py --phase-space --trajectories
```

---

## ❓ Preguntas Frecuentes

**Q: ¿Por qué solo 36k partículas de 1M simuladas?**  
A: El resto fue absorbida/dispersada fuera del campo o no alcanzó el plano.

**Q: ¿Es realista la geometría?**  
A: Sí, `elektaversa` está validado contra datos experimentales del fabricante.

**Q: ¿Puedo simular otros linacs?**  
A: Sí, hay modelos de Varian, Siemens en `opengate.contrib.linacs`

**Q: ¿Cuánto tarda una simulación real?**  
A: Para cálculo de dosis clínico: 1-8 horas (depende de precisión requerida)

**Q: ¿Cómo se compara con sistemas comerciales?**  
A: Monte Carlo es el gold standard. Sistemas como Eclipse, Monaco, RayStation usan MC o aproximaciones.

---

**Proyecto Modular 3 - CUCEI**  
*Simulación Monte Carlo de Radioterapia*  
Febrero 2026
