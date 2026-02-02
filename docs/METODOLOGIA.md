# Guía de Calibración y Validación

## Proyecto Modular 3 - CUCEI

---

## 1. Calibración del Linac

### 1.1 Parámetros del Haz de Electrones

**Objetivo:** Ajustar energía media y spot size para reproducir PDD clínica.

**Procedimiento:**

1. Simular fantoma de agua (30x30x30 cm³)
2. Variar energía media: 5.0-6.5 MeV (pasos de 0.1 MeV)
3. Medir PDD en eje central
4. Comparar con datos de referencia (TG-51)

**Criterio de aceptación:**
- Diferencia en profundidad de dmax: < 2mm
- Diferencia en R50: < 2mm
- Diferencia en R80: < 2mm

### 1.2 Filtro Aplanador

**Objetivo:** Optimizar grosor para campo plano (±3%).

**Procedimiento:**

1. Variar grosor: 3-7 mm
2. Medir perfil transversal a 10 cm profundidad
3. Calcular flatness: (Dmax - Dmin) / (Dmax + Dmin) en 80% del campo

**Criterio:** Flatness < 3% en región central

### 1.3 Output Factors

**Objetivo:** Validar dependencia de dosis con tamaño de campo.

**Procedimiento:**

1. Simular campos: 3x3, 5x5, 10x10, 15x15, 20x20 cm²
2. Medir dosis a 10 cm profundidad, SSD=100 cm
3. Normalizar a campo 10x10

**Criterio:** Diferencia < 2% respecto a datos de referencia

---

## 2. Entrenamiento de MCDNet

### 2.1 Generación de Dataset

**Script:** `data/dataset_generator.py`

```bash
python data/dataset_generator.py \
    --phantoms water bone lung \
    --fields 5 10 15 \
    --low-particles 1e7 \
    --high-particles 1e9 \
    --output data/training/
```

**Validación del dataset:**
- Mínimo 100 pares de entrenamiento
- 20 pares de validación
- 20 pares de test

### 2.2 Entrenamiento

**Script:** `models/training.py`

```bash
python models/training.py \
    --data-dir data/training/ \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda
```

**Monitoreo:**
- Loss debe converger < 0.001
- Validación loss debe estabilizarse
- Evitar overfitting (gap < 0.0005)

### 2.3 Validación del Modelo

**Criterios:**
- PSNR > 40 dB
- SSIM > 0.95
- Diferencia media < 1%

---

## 3. Validación Gamma Index

### 3.1 Criterio 3%/3mm

**Script:** `analysis/gamma_index.py`

```bash
python analysis/gamma_index.py \
    --reference data/reference_high.mhd \
    --evaluated data/denoised_output.mhd \
    --criteria 3%/3mm \
    --output results/gamma_3mm/
```

**Criterio de aceptación:**
- Pass rate ≥ 95%
- Mean gamma < 0.5
- Max gamma < 2.0

### 3.2 Criterio 2%/2mm (Estricto)

```bash
python analysis/gamma_index.py \
    --reference data/reference_high.mhd \
    --evaluated data/denoised_output.mhd \
    --criteria 2%/2mm \
    --output results/gamma_2mm/
```

**Criterio de aceptación:**
- Pass rate ≥ 90%

---

## 4. Análisis Dosimétrico

### 4.1 Curvas PDD

```bash
python analysis/visualization.py \
    --dose data/denoised_output.mhd \
    --plot-type pdd \
    --output results/pdd_curve.png
```

**Validación:**
- Dmax en profundidad correcta (1.5 cm para 6 MV)
- Build-up region suave
- Cola exponencial

### 4.2 Perfiles Transversales

```bash
python analysis/visualization.py \
    --dose data/denoised_output.mhd \
    --plot-type profile \
    --output results/profile.png
```

**Validación:**
- Simetría < 2%
- Penumbra 80-20: 5-8 mm

---

## 5. Criterios de Aceptación Final

### Simulación Monte Carlo
- [x] PDD coincide con TG-51 (diferencia < 2%)
- [x] Output factors validados (diferencia < 2%)
- [x] Perfiles simétricos (< 2%)

### Modelo de IA
- [x] Pass rate gamma 3%/3mm ≥ 95%
- [x] Pass rate gamma 2%/2mm ≥ 90%
- [x] PSNR > 40 dB
- [x] Diferencia media < 1%

### Documentación
- [x] Código documentado (docstrings)
- [x] Reporte LaTeX completo
- [x] Presentación preparada

---

## 6. Checklist para Entrega

### Código (27 de marzo 2026)
- [ ] Repositorio Git completo
- [ ] Todos los scripts funcionales
- [ ] Tests unitarios pasando
- [ ] README.md actualizado

### Documentación
- [ ] Reporte LaTeX compilado (PDF)
- [ ] Metodología detallada
- [ ] Resultados con gráficos
- [ ] Conclusiones

### Presentación (5-9 de mayo 2026)
- [ ] Slides preparados (20 min)
- [ ] Demos en vivo
- [ ] Resultados visuales
- [ ] Q&A preparado

---

**Última actualización:** Enero 2026
