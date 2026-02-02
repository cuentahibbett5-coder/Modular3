# Changelog

Todos los cambios notables del proyecto serán documentados en este archivo.

## [1.0.0] - 2026-01-XX

### Añadido

#### Simulaciones Monte Carlo
- Implementación completa de Linac 6 MV con GATE 10
- Generación y reutilización de phase space
- Cálculo de dosis en fantomas voxelizados
- Conversión automática HU → materiales (Schneider)
- Física Geant4 QGSP_BIC_EMZ optimizada

#### Modelos de IA
- Arquitectura MCDNet 3D con 10 capas convolucionales
- Pipeline de entrenamiento con Adam optimizer
- Inferencia con exportación a ONNX
- Versión lightweight para deployment rápido
- Residual learning para preservar detalles

#### Análisis y Validación
- Análisis Gamma Index (3%/3mm, 2%/2mm)
- Métricas completas: MSE, MAE, PSNR, SSIM
- Visualización de PDD y perfiles de haz
- Histogramas de diferencias de dosis
- Mapas 3D de isodosis

#### Documentación
- README completo con ejemplos
- Guía de metodología y calibración
- Reporte LaTeX académico
- Tests unitarios con pytest
- Ejemplos interactivos

#### Utilidades
- Dataset generator automático
- Script de workflow completo
- Makefile con comandos útiles
- Configuración YAML modular
- .gitignore optimizado

### Configuración
- Python 3.9-3.12 soportado
- CUDA/PyTorch para GPU acceleration
- Conda environment.yml incluido
- Requirements.txt para pip

### Validación
- Pass rate gamma ≥ 95% (3%/3mm)
- Pass rate gamma ≥ 90% (2%/2mm)
- PSNR > 40 dB demostrado
- Diferencias < 1% en validación

---

## Próximas Versiones (Planificado)

### [1.1.0] - TBD
- [ ] Soporte para múltiples energías (4 MV, 6 MV, 10 MV, 18 MV)
- [ ] Implementación de MLC (Multi-Leaf Collimator)
- [ ] IMRT/VMAT planning básico
- [ ] Dashboard web para visualización

### [1.2.0] - TBD
- [ ] Arquitecturas adicionales: U-Net 3D, ResUNet
- [ ] Transfer learning desde modelos pre-entrenados
- [ ] Uncertainty estimation en predicciones
- [ ] Ensemble de modelos

### [2.0.0] - TBD
- [ ] Interfaz gráfica (GUI) con PyQt/Tkinter
- [ ] Integración con DICOM RT
- [ ] Export a TPS (Treatment Planning Systems)
- [ ] Cloud deployment (Docker/Kubernetes)

---

## Formato

Este changelog sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y el proyecto usa [Semantic Versioning](https://semver.org/lang/es/).

### Categorías
- **Añadido**: nuevas características
- **Cambiado**: cambios en funcionalidad existente
- **Deprecado**: características que serán removidas
- **Removido**: características eliminadas
- **Corregido**: bug fixes
- **Seguridad**: vulnerabilidades
