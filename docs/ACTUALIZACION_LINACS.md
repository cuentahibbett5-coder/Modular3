# 🔄 Actualización del Proyecto - Usando Linacs Pre-Configurados

## ✅ **Cambios Realizados**

Hemos actualizado el proyecto para usar **linacs pre-configurados de OpenGate** en lugar de construir geometrías desde cero. Esto hace el código más:

- ✅ **Realista** - Geometrías validadas experimentalmente
- ✅ **Simple** - Menos código, más funcionalidad
- ✅ **Mantenible** - No reinventamos la rueda
- ✅ **Profesional** - Usamos estándares de la comunidad

---

## 📦 **Linacs Disponibles**

### **Elekta Versa** (Recomendado)
```python
import opengate.contrib.linacs.elektaversa as versa
linac = versa.add_linac(sim)
```
- Energías: 4MV, 6MV, 10MV, 15MV, 18MV
- MLC Agility (160 láminas, 5mm)
- Muy común en hospitales

### **Varian TrueBeam**
```python
import opengate.contrib.linacs.varian as varian
linac = varian.add_linac(sim)
```
- Energías múltiples
- MLC Millennium/HD120
- Estándar mundial

---

## 🚀 **Uso Simplificado**

### **1. Simulación Básica del Linac**

```python
from simulations.linac_6mv import LinacSimulation

# Crear con Elekta Versa pre-configurado
sim = LinacSimulation(
    linac_type='versa',  # ← Geometría real validada
    energy='6MV',
    field_size=(10, 10)
)

# ¡Una sola línea configura todo!
sim.setup_linac()  

# Física y ejecución
sim.setup_physics()
sim.add_phase_space_actor('data/ps.root')
sim.run(n_particles=1e8)
```

### **2. Generar Phase Space**

```bash
# Usando línea de comandos
python simulations/linac_6mv.py \
    --linac versa \
    --energy 6MV \
    --field 10 10 \
    --particles 1e8 \
    --output data/phase_space/versa_6mv.root
```

```python
# O desde Python
from simulations.phase_space import PhaseSpaceGenerator

gen = PhaseSpaceGenerator(linac_type='versa', energy='6MV')
gen.generate(
    output_path='data/phase_space/versa_6mv.root',
    num_particles=1e8,
    field_size=(10, 10)
)
```

### **3. Calcular Dosis (Super Simple)**

```python
from simulations.dose_calculation_simple import calculate_dose_in_water

# ¡Una función lo hace todo!
calculate_dose_in_water(
    linac_type='versa',
    energy='6MV',
    field_size=(10, 10),
    n_particles=1e7,
    output_path='dose.mhd'
)
```

O desde terminal:
```bash
python simulations/dose_calculation_simple.py \
    --linac versa \
    --energy 6MV \
    --field 10 10 \
    --particles 1e7 \
    --output data/dose_maps/dose.mhd
```

---

## 📊 **Comparación: Antes vs Ahora**

### **Antes (Geometría Manual)**
```python
# ~400 líneas de código construyendo:
- Target de tungsteno
- Colimador primario  
- Filtro aplanador
- Cámara de monitoreo
- Jaws (mordazas)
- Fuente de electrones
# ... parámetros aproximados
```

### **Ahora (Linac Pre-Configurado)**
```python
# ~250 líneas totales
linac = versa.add_linac(sim)  # ← TODO incluido, validado!
versa.set_default_source(sim, linac, '6MV')
```

**Ventajas del nuevo enfoque:**
- ✅ Geometría **exacta** del Elekta Versa real
- ✅ Parámetros **calibrados** con datos experimentales
- ✅ Fuente de electrones **optimizada** automáticamente
- ✅ Validado por la **comunidad OpenGate**

---

## 🔧 **Compatibilidad**

### **Si OpenGate tiene linacs pre-configurados:**
```python
✅ Usa Elekta Versa o Varian (automático)
```

### **Si NO están disponibles:**
```python
⚠️  Fallback a geometría mínima simplificada
   (Solo target de tungsteno básico)
```

El código detecta automáticamente qué está disponible:
```python
try:
    import opengate.contrib.linacs.elektaversa as versa
    VERSA_AVAILABLE = True
except ImportError:
    VERSA_AVAILABLE = False
    # Usa fallback
```

---

## 📚 **Archivos Actualizados**

| Archivo | Estado | Descripción |
|---------|--------|-------------|
| `linac_6mv.py` | ✅ Actualizado | Usa `versa.add_linac()` |
| `phase_space.py` | ✅ Actualizado | Simplificado con linacs reales |
| `dose_calculation_simple.py` | ✅ Nuevo | API super simple para dosis |
| `dose_calculation.py` | 📦 Backup | Original guardado como `.backup` |

---

## 🎯 **Workflow Recomendado**

```bash
# 1. Generar phase space (UNA VEZ)
python simulations/linac_6mv.py \
    --linac versa --energy 6MV --particles 1e8 \
    --output data/phase_space/versa_6mv.root

# 2. Calcular dosis baja estadística (RÁPIDO)
python simulations/dose_calculation_simple.py \
    --linac versa --energy 6MV --particles 1e7 \
    --output data/dose_maps/dose_noisy.mhd

# 3. Calcular dosis alta estadística (LENTO pero limpio)
python simulations/dose_calculation_simple.py \
    --linac versa --energy 6MV --particles 1e9 \
    --output data/dose_maps/dose_clean.mhd

# 4. Entrenar MCDNet con los pares (noisy, clean)
python models/training.py --data-dir data/training/

# 5. Aplicar denoising
python models/inference.py \
    --model models/checkpoints/mcdnet_best.pth \
    --input dose_noisy.mhd \
    --output dose_denoised.mhd

# 6. Validar con gamma
python analysis/gamma_index.py \
    --reference dose_clean.mhd \
    --evaluated dose_denoised.mhd
```

---

## 🌟 **Por Qué es Mejor**

### **Linacs Reales Validados**
Los modelos pre-configurados están basados en:
- Documentación técnica oficial (Elekta/Varian)
- Validaciones Monte Carlo publicadas
- Datos de commissioning de hospitales reales
- Testing extensivo de la comunidad OpenGate

### **Menos Errores**
- No hay riesgo de errores en geometría manual
- Parámetros ya optimizados (energía, spot size, filtros)
- Comportamiento predecible y reproducible

### **Más Profesional**
- Usas el estándar de la industria
- Código más corto y claro
- Fácil de mantener y extender

---

## 📖 **Referencias**

- **OpenGate Documentation**: https://opengate.readthedocs.io/
- **OpenGate contrib.linacs**: Pre-configured clinical linacs
- **Elekta Versa**: Modern linac (2010+) widely used
- **Validation**: Geometries validated against experimental data

---

## ⚡ **Migración Rápida**

Si tenías código viejo, actualiza así:

### Antes:
```python
from simulations.linac_6mv import LinacSimulation

sim = LinacSimulation(energy_MeV=5.8, spot_size_mm=3.0)
sim.setup_geometry()  # Construye todo manualmente
sim.setup_electron_source()
```

### Ahora:
```python
from simulations.linac_6mv import LinacSimulation

sim = LinacSimulation(linac_type='versa', energy='6MV')
sim.setup_linac()  # ¡Listo! Linac real configurado
```

---

**Fecha de actualización:** Febrero 2026  
**Proyecto Modular 3 - CUCEI**
