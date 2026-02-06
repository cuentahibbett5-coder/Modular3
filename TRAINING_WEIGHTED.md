# Entrenamiento con Máscara Ponderada y Filtrado de PDD Baja

## 🎯 Objetivo

Mejorar el modelo de denoising enfatizando la región clínicamente relevante (core) mientras se reduce el overfitting en zonas de ruido de Monte Carlo (periferia).

## 📊 Estrategia Implementada

### 1. **Filtrado de PDD Baja (1%)**
- **Qué es PDD**: Percent Depth Dose = máxima dosis por capa de profundidad
- **Acción**: Eliminar el 1% inferior de capas (dosis más baja)
- **Efecto**: Descarta capas donde el ruido monte carlo domina completamente
- **Beneficio**: Modelo no intenta aprender patrones ruidosos sin estructura real

### 2. **Pérdida Ponderada por Dosis**
```
Loss = mean((pred - target)² × weights)

donde:
  weights = 1.0  si dosis ≥ 20% del máximo (CORE)
  weights = 0.5  si dosis <  20% del máximo (PERIFERIA)
```

- **Core** (dosis alta): Peso total → Aprende bien la estructura principal
- **Periferia** (dosis baja): Peso reducido → No penaliza tanto los errores
- **Equilibrio**: Modelo aprende core perfectamente, ignora gracefully periferia ruidosa

---

## 🔧 Implementación

### Script Principal: `train_weighted.py`

```python
# Dataset con filtrado PDD
train_ds = SimpleDoseDatasetWeighted(
    TRAIN_DIR, 
    DATASET_ROOT, 
    INPUT_LEVELS,
    PATCH_SIZE,
    percentile_pdd=1.0,  # Eliminar 1% inferior
    is_train=True
)

# Loss ponderado
def weighted_mse_loss(pred, tgt, weights):
    diff = (pred - tgt) ** 2
    weighted_diff = diff * weights
    return weighted_diff.mean()
```

### Características Clave:

✅ **Cálculo automático de PDD**
```python
pdd = np.array([np.max(vol[z]) for z in range(D)])
```

✅ **Máscara de dosis**
```python
weights = np.where(dose ≥ threshold, 1.0, 0.5)
```

✅ **Compatible con normalización por máximo de input**
```python
# Normalizar AMBOS (input y target) por max(input)
# Permite modelo aprender amplificación de dosis
```

---

## 📋 Parámetros de Configuración

```python
DOSE_THRESHOLD = 0.20      # 20% del máximo = límite core/periferia
LOW_WEIGHT     = 0.5       # Peso para periferia (< 20%)
HIGH_WEIGHT    = 1.0       # Peso para core (≥ 20%)
PERCENTILE_PDD_LOW = 1.0   # Eliminar 1% inferior
```

Estos parámetros pueden ajustarse en `train_weighted.py` según sea necesario.

---

## 🚀 Ejecución

### Local (si hay GPU):
```bash
python train_weighted.py
```

### En Cluster (SLURM):
```bash
sbatch run_train_weighted.sh
```

El script creará checkpoints en `runs/denoising_weighted/`

---

## 📊 Evaluación y Comparación

### Comparar modelos:
```bash
python compare_models.py \
    --simple runs/denoising_v2/best.pt \
    --weighted runs/denoising_weighted/best.pt \
    --output comparison.png
```

Genera figura con:
- Época en que se alcanzó mejor loss
- Valor de val_loss para cada modelo
- Análisis cuantitativo de mejora

### Evaluar con normalización correcta:
```bash
python eval_correct_norm.py
```

Muestra:
- MAE, RMSE, correlación por sample
- Ratio Pred/Target (¿aprende a amplificar?)
- Tabla comparativa de predicciones

---

## 🎓 Interpretación de Resultados

### Si val_loss(weighted) < val_loss(simple):
✅ **El modelo weighted es mejor**
- Aprende mejor la estructura principal (core)
- Menos distorsionado por ruido periférico
- Mejor generalización a nuevos datos

### Comportamiento esperado:
- **Train loss**: Puede ser un poco más alto (pero centrado en voxeles importantes)
- **Val loss**: Debería ser **significativamente menor**
- **Visualmente**: Predicciones más limpias en core, periferia más ruidosa pero esperado

---

## 🔍 Diferencias clave vs Simple

| Aspecto | Simple | Weighted |
|---------|--------|----------|
| **PDD Filtering** | No | Sí (elimina 1% inferior) |
| **Loss Weighting** | Uniforme | Ponderado (core >> periferia) |
| **Robustez** | Sensible a ruido periférico | Resistente a ruido |
| **Core Quality** | Buena | **Mejor** |
| **Periferia** | Intenta aprender ruido | Descarta gracefully |
| **Generalización** | Limitada | **Mejorada** |

---

## 📈 Próximos Pasos

1. **Entrenar modelo weighted** en cluster
2. **Comparar val_loss** con modelo simple
3. **Evaluar predicciones** con `eval_correct_norm.py`
4. **Analizar visualización** de slices (¿mejor denoising en core?)
5. **Ajustar parámetros** si es necesario:
   - Aumentar `LOW_WEIGHT` (0.5 → 0.7) si periferia está completamente basura
   - Cambiar `DOSE_THRESHOLD` (0.20 → 0.30) si queremos core más grande
   - Cambiar `PERCENTILE_PDD_LOW` (1.0 → 2.0) si queremos descartar más capas

---

## ⚠️ Notas Importantes

- **El modelo weighted PENALIZA MENOS los errores en periferia**
  - Esto no es "trampa", es reconocer que periferia es principalmente ruido monte carlo
  - Diferente a IGNORAR, sigue aprendiendo pero con menos peso

- **Los pesos se calculan dinámicamente por patch**
  - Cada batch obtiene pesos basados en su máximo local de dosis
  - Flexible y adaptativo

- **Compatible con normalización correcta**
  - Mantiene `max(input)` normalization
  - Así modelo aprende amplificación: small_input → small_output, large_input → large_output

---

## 📂 Archivos Relacionados

- `train_weighted.py` - Script principal de entrenamiento
- `run_train_weighted.sh` - Script SLURM para cluster
- `compare_models.py` - Herramienta de comparación
- `eval_correct_norm.py` - Evaluación con normalización correcta
- `GT_LAYER_ANALYSIS.md` - Análisis capa-por-capa que motivó esto

---

**Status**: ✅ Ready to train

**Next**: Execute `sbatch run_train_weighted.sh` en cluster

