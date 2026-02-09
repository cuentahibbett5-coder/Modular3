# Verificación de Multiplicador Trivial - Instrucciones para Cluster

## ¿Qué verifica este análisis?

**Pregunta clave:** ¿Es realmente inteligente el modelo o solo hace `Predicción = Input × factor_constante`?

Si fuera solo un multiplicador:
- ❌ **No sería útil** vs simular directamente más eventos 
- ❌ **Desperdicio computacional** - mejor usar más tiempo de simulación
- ❌ **Falsa innovación** - no aporta valor real

## Ejecución en Cluster

### Prerrequisitos
```bash
# 1. Datos exportados (generar si no existen)
python export_predictions.py

# 2. Verificar estructura
ls exports/
# Debe contener: *_input.npy, *_pred.npy, *_target.npy
```

### Comando Principal
```bash
# Análisis completo
python verify_cluster_multiplier.py

# Con opciones
python verify_cluster_multiplier.py \
  --output-dir multiplicador_analysis \
  --max-cases 5
```

### Outputs Generados

1. **`multiplier_analysis_summary.json`** - Métricas detalladas
2. **`{case}_trivial_analysis.png`** - Visualización por caso
3. **Terminal**: Veredicto final con explicación

## Criterios de Verificación

### 🚨 Modelo TRIVIAL si:
- Correlación con `input × factor` > 0.98
- Diferencia normalizada < 1% 
- Sin mejora vs multiplicador simple
- Factor espacialmente uniforme (CV < 5%)

### ✅ Modelo INTELIGENTE si:
- Mejora > 3x vs input ruidoso
- Baja correlación con multiplicador < 0.95
- Variación espacial significativa
- Aprendizaje de patrones complejos

## Interpretación de Resultados

### Caso A: Multiplicador Trivial
```
⚠️ LA MAYORÍA SON MULTIPLICADORES TRIVIALES
   → El modelo no es mejor que simular más eventos
   → Revisar arquitectura y entrenamiento
```
**Acción:** Cambiar función de pérdida, arquitectura o datos

### Caso B: Modelo Inteligente  
```
✅ EL MODELO ES GENUINAMENTE INTELIGENTE
   → Va más allá del simple escalado
   → Útil para denoising de dosis
```
**Acción:** Documentar y publicar resultados

### Métricas Clave en JSON

```json
{
  "analysis_summary": {
    "trivial_cases": 0,           // ← Debe ser bajo
    "useful_cases": 8,            // ← Debe ser alto
    "avg_improvement_vs_input": 4.2,  // ← >2.0 es bueno
    "avg_correlation_with_naive": 0.85 // ← <0.95 es bueno
  }
}
```

## Troubleshooting

### Error: No se encontraron datos
```bash
# Generar primero los exports
python export_predictions.py
```

### Error: No se encontró modelo
```bash
# Verificar modelos disponibles
find . -name "*.pt" | head -5
```

### Sin plots generados
```bash
# Verificar matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# Debe ser 'Agg' para cluster
```

## Validación Clínica

### Factor de Escalado Esperado
- **1M → 29M eventos**: Factor ≈ 29
- **Variación natural**: ±10% por anatomía
- **Si CV > 20%**: Modelo aprende patrones espaciales

### Límites de Aceptación
- **Mejora mínima**: 1.5x vs input
- **Correlación máxima con ingenua**: 0.95
- **MAE normalizado máximo**: 0.05

### Red Flags 🚨
- Factor uniforme en todo el volumen
- Correlación perfecta (>0.99) con input×constante  
- Sin mejora en regiones de alta dosis (críticas)

## Extensiones Futuras

1. **Análisis por región anatómica**
2. **Validación con diferentes niveles de ruido** 
3. **Comparación con modelos benchmark**
4. **Métricas específicas dosimétricas (DVH)**

---

**Objetivo Final**: Demostrar que el modelo AI aporta valor real más allá del simple escalado de eventos, justificando su uso clínico vs simulaciones más largas.