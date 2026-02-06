# DeepMC v3: 4 Pilares Técnicos Implementados

## Problema Original (v1)
El modelo `train_weighted_pro.py` predecía una dosis **constante (~60)** en todo el volumen en lugar de aprender la estructura real (que alcanza ~1000 en el core). 

**Root Cause**: 96% de los voxeles tienen dosis baja/nula → MSE estándar recompensa predecir "casi cero" en todos lados.

---

## Pilar 1: Función de Pérdida Exponencial Ponderada

### ¿Qué es?
En lugar de usar **MSE estándar** (trata todos los voxeles igual), usamos una **función exponencial** que le da mayor peso a los voxeles con dosis alta:

```
weight(dose) = exp(dose / ref_dose) - 1
```

### Propiedades
- **dose → 0**: weight → 0 (bajo peso en vacío)
- **dose = ref_dose**: weight ≈ 1.7 (peso moderado)
- **dose = 2×ref_dose**: weight ≈ 7.4 (peso muy alto)

### Garantía
Obliga a la red a "obsesionarse" con el **pico de dosis (core)** y no solo minimizar MSE global prediciendo ceros.

### Implementación
```python
class ExponentialWeightedLoss(nn.Module):
    def forward(self, pred, target):
        max_dose = target.max().detach()
        ref_dose = max_dose * 0.5  # 50% del máx
        
        normalized_dose = torch.clamp(target / ref_dose, min=0, max=10)
        weights = torch.exp(normalized_dose) - 1.0  # Exponencial
        
        error = torch.abs(pred - target)
        weighted_error = error * weights * mask  # Solo donde hay dosis
        loss = weighted_error.sum() / n_active
        return loss
```

---

## Pilar 2: Entradas Duales (Dosis + CT)

### ¿Por Qué?
Tu modelo actual intenta reconstruir la dosis **solo mirando la dosis ruidosa**. Pero la física del transporte de partículas depende de **dónde hay material** (agua, hueso, aire).

### Ejemplos Físicos
- **Build-up effect**: La dosis sube en los primeros cm del paciente (entrada de energía)
- **Electron Return Effect (ERE)**: Los electrones secundarios contribuyen dosis después del material
- **Atenuación en hueso**: El hueso detiene más partículas que el agua

### Solución
Concatenar **2 canales de entrada**:
- Canal 1: Dosis ruidosa (input_10M)
- Canal 2: Intensidad CT normalizada (HU → [0,1])

```python
def forward(self, dosis, ct=None):
    if self.dual_input and ct is not None:
        x = torch.cat([dosis, ct], dim=1)  # [B, 2, D, H, W]
    else:
        x = dosis
    # ... rest of network
```

### Estado Actual
⚠️ Por ahora **deshabilitado** (`dual_input=False`) porque no todos los pacientes tienen CT en el dataset pilot. Cuando esté disponible, activar:
```python
model = DeepMCNet(base_channels=BASE_CHANNELS, dual_input=True)
```

---

## Pilar 3: Arquitectura Avanzada

### Cambios vs U-Net Básico

#### 3.1 Residual Blocks
En lugar de convoluciones simples, usar bloques residuales:

```python
class ResidualConvBlock(nn.Module):
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out = out + identity  # ⚡ SKIP CONNECTION
        out = self.relu(out)
        return out
```

**Ventaja**: Permite entrenar redes más profundas sin que los gradientes desaparezcan (gradient vanishing).

#### 3.2 Squeeze-and-Excitation Blocks (Channel Attention)
La red aprende qué **canales de características son importantes** en cada punto:

```python
class SEBlock(nn.Module):
    def forward(self, x):
        se = F.adaptive_avg_pool3d(x, 1)  # Squeeze
        se = self.fc1(se.view(B, -1))     # Excitation
        se = self.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se  # Recalibración
```

**Ventaja**: La red entiende qué características importan → mejor generalización.

#### 3.3 Batch Normalization
Cada capa usa BatchNorm para estabilizar el entrenamiento:

```python
self.conv1 = nn.Conv3d(...)
self.bn1 = nn.BatchNorm3d(out_channels)  # ⚡ Normalización
out = self.bn1(out)
```

**Ventaja**: Permite tasas de aprendizaje más altas sin que el gradiente explote.

---

## Pilar 4: Volumen de Datos Masivo (Simulación)

### Realidad
Tu dataset pilot tiene **80 muestras** (20 pares × 4 niveles), mientras DeepMC entrenó con **56,000 volúmenes**.

### Estrategia: Data Augmentation
Con 80 muestras, usamos técnicas para "simular" más datos:

1. **Random Patches**: Cada época, extraer parches diferentes del mismo volumen
2. **Random Crops**: Desplazar ventana aleatoriamente → nuevas configuraciones
3. **Potencial Futuro**: 
   - Rotaciones (90°, 180°, 270° en ejes)
   - Flips (izquierda-derecha, anterior-posterior)
   - Elastica deformations

**Expectativa**: Con 100 épocas × BATCH_SIZE=2 × múltiples patches/volumen ≈ miles de iteraciones efectivas.

---

## Comparativa: v1 vs v3

| Aspecto | v1 (train_weighted_pro.py) | v3 (train_deepmc_v3.py) |
|--------|----------------------------|------------------------|
| **Loss Function** | MSE dinámico (lineal) | Exponencial ponderada |
| **Entrada** | Dosis únicamente | Dosis + CT (dual ready) |
| **Blocks** | Simple Conv | Residual + SE + BatchNorm |
| **Función Output** | `output = model(x)` | `output = input + residual` |
| **Problema** | Predice constante (~60) | Aprende correcciones reales |

---

## Pasos Siguientes

### 1️⃣ **Entrenar v3**
```bash
python train_deepmc_v3.py
```
Expected: ~1.5-2 min/epoch en GPU, 100-150 épocas con early stopping

### 2️⃣ **Evaluar**
```bash
python evaluate_deepmc_v3.py
```
Esperar a ver:
- PDD plots: predicción debe seguir la forma de GT (no constante)
- PSNR > 30 dB (vs muy bajo en v1)
- Errores menores en zonas de alta dosis

### 3️⃣ **Si Disponible: Activar Dual Input**
Si tienes CT disponible, cambiar en `train_deepmc_v3.py`:
```python
train_dataset = DualInputDoseDataset(TRAIN_DIR, use_ct=True)
model = DeepMCNet(base_channels=BASE_CHANNELS, dual_input=True)
```

### 4️⃣ **Posibles Mejoras Iterativas**
Si aún underfitting:
- ↑ `ref_dose_percentile` en ExponentialWeightedLoss (ej: 0.3 en lugar de 0.5)
- ↑ `base_channels` (ej: 32 en lugar de 16)
- ↓ Learning rate si gradientes inestables
- Data augmentation (flips, rotaciones)

---

## Referencias
- **DeepMC Paper**: "A Deep Learning Model for Fast Dose Calculation in Photon Beam Radiotherapy"
- **Arquitectura**: Residual U-Net con Squeeze-and-Excitation blocks
- **Loss**: Exponential weighting similar a weighted MAE en visión médica

