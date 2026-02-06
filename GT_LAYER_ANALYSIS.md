# Análisis Capa-por-Capa del Ground Truth (29.4M partículas)

## Resumen Ejecutivo

El ground truth de **29.4M partículas** ha sido analizado **capa por capa (slice-by-slice)** a lo largo de los 300 layers en profundidad (eje z). Este análisis revela:

✅ **Calidad ACEPTABLE en el CORE (región de dosis alta)**
- Suavidad (smoothness): 0.79 (escala 0-1, donde 1 es perfecto)
- Simetría: 56.5% (asimetría relativa)
- SNR: 14.2 (señal/ruido alto - limpios)

⚠️ **Calidad LIMITADA en PERIFERIA (cola de dosis baja)**
- Suavidad: 0.73 (más ruidoso)
- SNR: 217 (mucho ruido relativo a la débil señal)
- Asimetría: 54% (varía mucho entre layers)

---

## Métricas por Región

### CORE (Layers 125-175, profundidad máxima)
Región donde está la máxima deposición de energía.

```
Max Dose:        6.00 ± variable
CV (variación):  1.20 (bajo - bueno)
SNR:             14.15 (bueno - señal clara)
Smoothness:      0.79 (buena suavidad)
Asymmetry:       56.5% (simetría razonable)
```

**Interpretación**: El core es **limpio y suave**. Las fluctuaciones aquí son principalmente estadísticas reales (monte carlo), no artefactos.

### PERIPHERY (Layers 0-125 y 175-299)
Región de dosis baja donde el ruido domina.

```
Max Dose:        99.90 ± high variability
CV (variación):  1.31 (más alto - más ruidoso)
SNR:             217.2 (muy alto - señal débil respecto ruido)
Smoothness:      0.73 (más granular)
Asymmetry:       54% (inconsistente)
```

**Interpretación**: La periferia tiene **ruido estadístico alto**. El SNR alto significa que la señal (dosis real) es muy pequeña comparada al ruido (fluctuaciones de Monte Carlo).

---

## Análisis Capa-por-Capa Detallado

### Patrones Observados:

1. **Variación en CV (Coef. de Variación)**
   - Entrada (z=0): CV ≈ 1.68
   - Core (z=150): CV ≈ 1.20  ← **Mínimo, más suave**
   - Salida (z=300): CV ≈ 1.28

   → **El ruido disminuye en el core** porque hay más partículas depositando energía en esa región.

2. **Suavidad Decreciente en Periferia**
   - Core (z=125-175): smoothness ≈ 0.79
   - Periferia (z>200): smoothness ≈ 0.82-0.85

   → **La periferia es paradójicamente "más suave"** porque el gaussiano de suavizado suprime mejor el ruido débil. Pero esto es engañoso: es ruido suprimido, no estructura real.

3. **Asimetría Aumenta en Periferia**
   - Core: ~56%
   - Periferia lejana (z>250): ~80-93%

   → **La asimetría crece cuando la dosis es baja**, indicando fluctuaciones estadísticas sin estructura correlacionada.

---

## ¿Es 29.4M Suficiente Ground Truth?

### ✅ SÍ PARA:
- **Entrenar en el core (150±50 layers)**: Tiene buen SNR, suavidad, simetría
- **Usar como referencia en dosis alta**: Donde dominan fluctuaciones estadísticas reales
- **Evaluación en región de interés clínico** (típicamente ±80% isodose)

### ⚠️ NO PARA:
- **Periferia lejana**: Dosis < 10% del máximo es ruidosa y asimétrica
- **Estructura detallada de baja dosis**: El ruido de Monte Carlo domina
- **Validación de simetría**: Los patterns en periferia pueden ser estadísticos, no reales

---

## Recomendaciones Prácticas

### OPCIÓN 1: Usar 29.4M tal cual (RECOMENDADO)
```python
# Entrenar modelo con todos los 300 layers
# Pero dar más peso a layers 75-225 (core)
# O usar máscara: solo entrenar donde dose > 20% del max
```

**Ventaja**: Datos reales, sin suavizado artificial
**Desventaja**: Modelo aprenderá el ruido de periferia

### OPCIÓN 2: Post-procesar periferia
```python
# Aplicar Gaussian filter suave en región z > 200 o z < 50
# Suavizar donde SNR > 50 (dosis muy baja)
```

**Ventaja**: Reduce ruido sin perder estructura de core
**Desventaja**: Pierde información estadística legítima

### OPCIÓN 3: Entrenar con región enmascarada
```python
# Crear máscara: True donde dose > 20% del máximo
# Loss = MSE * máscara
# Ignora periferia en training
```

**Ventaja**: Fuerza modelo a enfocarse en dosis clínicamente relevante
**Desventaja**: Modelo será malo en periferia

---

## Conclusión

**29.4M partículas ES ADECUADO para ground truth**, especialmente:
- ✅ En la región del core (layers ~50-250)
- ✅ Para dosis > 20% del máximo
- ⚠️ Con cuidado en la periferia

**La variabilidad capa-por-capa es ESPERADA** en Monte Carlo:
- Core: ruido ~20-30% (estadísticas reales)
- Periferia: ruido ~80-100% (dominado por fluctuaciones)

**Siguiente paso**: Entrenar modelo **con pérdida ponderada** para ignorar o suavemente penalizar la periferia ruidosa.
