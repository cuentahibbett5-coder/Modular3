# 📊 DeepMC v3: Resumen Visual de Cambios

## 🔴 Problema de v1 en Una Imagen

```
v1: MSE Standard (96% de datos son cero)

┌─────────────────────────────────────────────────────┐
│  Distribución de Voxeles en Volumen                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  96% CEROS (ruido fondo):  ████████████████████  │
│  4% SEÑAL (dosis real):    ██                     │
│                                                      │
└─────────────────────────────────────────────────────┘

Model MSE Loss:
  = mean((pred - target)²)
  ≈ mostly_zeros  ← DOMINA EL CÁLCULO
  
Conclusión trivial del optimizer:
  "Predecir ~0 en todas partes minimiza MSE global"
  
Resultado:
  ❌ Predicción plana (~60)
```

---

## 🟢 Solución de v3: Loss Exponencial

```
v3: Exponential Loss (pesa voxeles con dosis)

weight(dose) = exp(dose/ref) - 1

┌─────────────────────────────────────────────────────┐
│  Pesos Aplicados por Dosis                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│  dose = 0:      weight ≈ 0.0      (bajo)         │
│  dose = 500:    weight ≈ 1.7      (moderado)     │
│  dose = 1000:   weight ≈ 7.4      (ALTO)         │
│                                                      │
│  Core (1000): ✕7.4  ← MULTIPLICA EL ERROR        │
│  Ruido (0):  ✕0.0   ← IGNORA PRÁCTICAMENTE       │
│                                                      │
└─────────────────────────────────────────────────────┘

Conclusión del optimizer:
  "Errar en el core cuesta 7.4× más"
  
Resultado:
  ✅ Predicción estructura real
```

---

## 🏗️ Cambios Arquitectónicos

### v1: U-Net Básico
```
Input → Conv → Conv → Conv → ... → Output
        
Problemas:
- Gradientes desaparecen en capas profundas
- No hay mecanismo de atención (todos los canales igual)
- Sin normalización (entrenamiento inestable)
```

### v3: DeepMC-Style
```
Input [D+CT]  
  ↓
┌─────────────────────────┐
│ Encoder (Residual Blocks)│  ← Preservan gradientes
│ + SE Blocks             │  ← Atención por canal
│ + Batch Norm            │  ← Estabilización
└─────────────────────────┘
  ↓
Bottleneck
  ↓
┌─────────────────────────┐
│ Decoder (Residual Blocks)│
│ + SE Blocks             │
│ + Batch Norm            │
└─────────────────────────┘
  ↓
Output = Input + Residual  ← Aprender correcciones
```

Ventajas:
- ✅ Gradientes fluyen sin desvanecerse
- ✅ Red aprende qué canales importan (SE blocks)
- ✅ Entrenamiento estable (batch norm)
- ✅ Aprender "delta" es más fácil (residual)

---

## 📊 Tabla Comparativa

| Aspecto | v1 | v3 |
|---------|----|----|
| **Loss** | MSE lineal | Exponencial |
| **Mask** | No | Sí (voxels > 0) |
| **Input Channels** | 1 | 2 (ready) |
| **Encoder Block** | Simple Conv | Residual + SE + BN |
| **Skip Connections** | Concat solo | Residual + Concat |
| **Output** | Absolute | Input + Residual |
| **Predicción Esperada** | Plana (~60) | Estructura (0-1000) |
| **PSNR Esperado** | <20 dB | >30 dB |

---

## 🎯 Flujo de Entrenamiento

```
Dataset (80 muestras × 4 niveles)
  ↓
Random patching (96³ voxeles)
  ↓
┌──────────────────────────────────────────┐
│ Epoch 1-10: Loss baja rápidamente        │
│ Modelo aprende estructura básica         │
└──────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────┐
│ Epoch 10-30: Mejora gradual              │
│ Fine-tuning de detalles                  │
└──────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────┐
│ Epoch 30-50: Plateau                     │
│ Early stopping se dispara (patience=20)  │
└──────────────────────────────────────────┘
  ↓
Best Model Saved
  ↓
Evaluation (PDD, PSNR, SSIM)
```

---

## 🔬 Hipótesis Detrás de Cada Pilar

### Pilar 1: Loss Exponencial
**Hipótesis**: El optimizer ignora el 96% de ruido si no se le da instrucciones

**Implementación**: Pesar voxeles por dosis

**Validación**: Si funciona, PDD debe mostrar estructura (no plana)

### Pilar 2: Entrada Dual
**Hipótesis**: Sin contexto geométrico, impossible reconstruir física

**Implementación**: Concatenar [Dosis, CT]

**Validación**: Con CT, error debe bajar 10-20%

**Status**: Deshabilitado (esperar dataset completo)

### Pilar 3: Arquitectura Avanzada
**Hipótesis**: U-Net básico no es suficiente para 96% ruido

**Implementación**: Residual + SE + BatchNorm

**Validación**: Training debe ser más estable (sin NaN)

### Pilar 4: Data Strategy
**Hipótesis**: 80 muestras es poco pero suficiente con augmentation

**Implementación**: Random patching + 100 épocas

**Validación**: Convergencia en 30-50 épocas (no overfitting rápido)

---

## 📈 Métricas de Éxito

### ✅ El Modelo Funciona Si:
1. **Val Loss**: Baja y se estabiliza (no diverge)
2. **PSNR**: > 30 dB (vs <20 en v1)
3. **SSIM**: > 0.85 (estructura preservada)
4. **PDD Shape**: Sigue GT (campaniforme, no plana)
5. **High Dose Error**: < Mid Dose < Low Dose
6. **Early Stopping**: Se activa ~epoch 30-50

### ❌ El Modelo Falla Si:
1. **Loss NaN**: Gradientes inestables
2. **PSNR**: < 20 dB (no mejora vs v1)
3. **PDD Plana**: Misma predicción constante
4. **No converge**: Loss sigue alto después de 50 épocas
5. **OOM**: Excede memoria GPU

---

## 🚀 Timeline Esperado

```
T+0h:    Iniciar entrenamiento
T+0.5h:  Epoch 1-5, val_loss baja
T+1h:    Epoch 15-20, empieza fine-tuning
T+1.5h:  Epoch 30-40, cerca del plateau
T+2h:    Epoch 45-50, early stopping
T+2.1h:  Guardar best_model.pt ✅
T+2.2h:  python evaluate_deepmc_v3.py
T+2.3h:  Resultados listos (PSNR, SSIM, PDD)
```

---

## 💡 Key Insight

**El problema NO era que el modelo fuera incapaz.**

El modelo es capaz de aprender. El problema era que **le dimos los incentivos equivocados** (MSE estándar).

v3 **cambia los incentivos** (loss exponencial) para que el modelo aprenda lo correcto.

Es un ejemplo perfecto de cómo en ML, **la función objetivo es crítica**:
- Objetivo incorrecto → Soluciones malas (v1 predice cero)
- Objetivo correcto → Soluciones buenas (v3 aprende estructura)

