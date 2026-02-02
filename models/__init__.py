"""
Paquete de modelos de Inteligencia Artificial para dosimetría

Módulos principales:
- mcdnet: Arquitectura MCDNet 3D para denoising de dosis
- training: Pipeline de entrenamiento de modelos
- inference: Inferencia y exportación ONNX
- utils: Utilidades y funciones auxiliares

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

__version__ = '1.0.0'

try:
    from .mcdnet import MCDNet3D
    from .training import MCDNetTrainer
    from .inference import DoseDenoiser
    
    __all__ = [
        'MCDNet3D',
        'MCDNetTrainer',
        'DoseDenoiser',
    ]
except ImportError as e:
    print(f"⚠ Advertencia: No se pudieron importar todos los módulos de IA")
    print(f"  Razón: {e}")
    print(f"  Instala PyTorch con: pip install torch torchvision")
    __all__ = []
