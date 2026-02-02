# 🎯 Ejemplos - Proyecto Modular 3

Este directorio contiene ejemplos prácticos de uso del proyecto.

## 📁 Ejemplos Disponibles

### 1. complete_workflow.py
**Ejemplo completo del workflow de simulación y denoising**

Demuestra el flujo completo:
1. Generación de phase space del linac
2. Cálculo de dosis con baja estadística
3. Aplicación de denoising con MCDNet
4. Validación con gamma index
5. Cálculo de métricas y visualización

**Uso:**
```bash
python examples/complete_workflow.py
```

**Outputs:**
- `results/example_noisy.mhd` - Dosis con ruido (baja estadística)
- `results/example_denoised.mhd` - Dosis denoised (limpia)
- `results/figures/comparison.png` - Comparación visual
- `results/figures/pdd_curve.png` - Curva PDD

---

## 🚀 Ejemplos por Módulo

### Simulación del Linac

```python
from simulations.linac_6mv import LinacSimulation

# Crear simulación
linac = LinacSimulation(
    energy_mean_MeV=5.8,
    energy_sigma_percent=3.0,
    spot_size_mm=3.0
)

# Configurar geometría
linac.setup_geometry()
linac.setup_electron_source()

# Añadir phase space actor
linac.add_phase_space_actor(
    output_path='data/phase_space/linac_ps.root',
    plane_position_mm=400
)

# Ejecutar
linac.run(n_particles=1e8)
```

### Generación de Phase Space

```python
from simulations.phase_space import PhaseSpaceGenerator

# Generar phase space
generator = PhaseSpaceGenerator(
    energy_mean_MeV=5.8,
    n_particles=1e8,
    output_path='data/phase_space/linac_ps.root'
)

generator.generate()

# Analizar
stats = generator.analyze_phase_space()
print(f"Partículas: {stats['n_particles']}")
print(f"Energía media: {stats['mean_energy']:.2f} MeV")
```

### Cálculo de Dosis

```python
from simulations.dose_calculation import DoseCalculator

# Crear calculador
calc = DoseCalculator(
    phase_space_path='data/phase_space/linac_ps.root',
    ct_image_path='data/phantoms/phantom.mhd'
)

# Configurar y calcular
calc.create_voxelized_phantom()
calc.add_dose_actor(output_path='results/dose.mhd')
calc.run(n_particles=1e7)
```

### Entrenamiento de MCDNet

```python
from models.training import MCDNetTrainer, DoseDataset
from models.mcdnet import create_mcdnet
from torch.utils.data import DataLoader

# Crear datasets
train_dataset = DoseDataset('data/training/train')
val_dataset = DoseDataset('data/training/val')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Crear modelo
model = create_mcdnet('standard')

# Entrenar
trainer = MCDNetTrainer(model, train_loader, val_loader)
trainer.train(epochs=100)
```

### Inferencia

```python
from models.inference import DoseDenoiser

# Cargar modelo
denoiser = DoseDenoiser('models/checkpoints/mcdnet_best.pth')

# Denoise archivo
denoiser.denoise_file(
    input_path='data/noisy_dose.mhd',
    output_path='results/denoised_dose.mhd'
)

# Exportar a ONNX
denoiser.export_to_onnx('models/mcdnet.onnx')
```

### Análisis Gamma

```python
from analysis.gamma_index import validate_dose_comparison

# Validar con gamma
results = validate_dose_comparison(
    ref_path='data/reference_dose.mhd',
    eval_path='results/denoised_dose.mhd',
    criteria='3%/3mm',
    output_dir='results/gamma/'
)

print(f"Pass rate: {results['pass_rate']:.2f}%")
```

### Visualización

```python
from analysis.visualization import (
    plot_dose_comparison,
    plot_pdd_curve,
    plot_beam_profile
)
import SimpleITK as sitk

# Cargar dosis
dose_image = sitk.ReadImage('results/denoised_dose.mhd')
dose = sitk.GetArrayFromImage(dose_image)

# PDD
plot_pdd_curve(dose, save_path='results/pdd.png')

# Perfil
plot_beam_profile(dose, depth_idx=100, axis='x', 
                 save_path='results/profile.png')
```

### Métricas

```python
from analysis.metrics import evaluate_all_metrics, print_metrics_report
import SimpleITK as sitk

# Cargar dosis
ref = sitk.GetArrayFromImage(sitk.ReadImage('data/reference.mhd'))
eval = sitk.GetArrayFromImage(sitk.ReadImage('results/denoised.mhd'))

# Evaluar
metrics = evaluate_all_metrics(ref, eval)

# Imprimir reporte
print_metrics_report(metrics)
```

---

## 📊 Ejemplo de Dataset Sintético

```python
import numpy as np
import SimpleITK as sitk

# Crear dosis sintética (gaussiana 3D)
size = (100, 100, 100)
z, y, x = np.indices(size)
center = np.array(size) / 2
sigma = 15

clean_dose = np.exp(-((x - center[2])**2 + 
                      (y - center[1])**2 + 
                      (z - center[0])**2) / (2 * sigma**2))

# Añadir ruido
noisy_dose = clean_dose + np.random.normal(0, 0.1, size) * clean_dose
noisy_dose = np.maximum(noisy_dose, 0)

# Guardar
clean_img = sitk.GetImageFromArray(clean_dose.astype(np.float32))
noisy_img = sitk.GetImageFromArray(noisy_dose.astype(np.float32))

sitk.WriteImage(clean_img, 'clean.mhd')
sitk.WriteImage(noisy_img, 'noisy.mhd')
```

---

## 🔧 Tips de Uso

### 1. Verificar instalación
```bash
python -c "import opengate, torch, SimpleITK; print('✅ Todo instalado')"
```

### 2. Ejecutar tests
```bash
pytest tests/ -v
```

### 3. Ver ayuda de scripts
```bash
python simulations/linac_6mv.py --help
python models/training.py --help
python analysis/gamma_index.py --help
```

### 4. Monitorear GPU
```bash
watch -n 1 nvidia-smi  # Durante entrenamiento
```

---

## 📝 Notas

- Los ejemplos usan datos sintéticos por defecto
- Para simulaciones reales, descomentar las líneas de ejecución
- Las simulaciones completas pueden tardar horas/días
- Se recomienda GPU para entrenamiento

---

## 🆘 Troubleshooting

**Error: "No module named 'opengate'"**
```bash
pip install opengate
```

**Error: "CUDA not available"**
```python
# Usar CPU en su lugar
trainer = MCDNetTrainer(model, train_loader, val_loader, device='cpu')
```

**Error: "Memory error"**
```python
# Reducir batch size
train_loader = DataLoader(dataset, batch_size=2)  # En vez de 4
```

---

¿Tienes más preguntas? Revisa la [documentación completa](../README.md) 📚
