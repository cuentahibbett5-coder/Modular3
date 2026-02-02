"""
Paquete de simulaciones de radioterapia con GATE 10 / OpenGate

Módulos principales:
- linac_6mv: Modelado de acelerador lineal de 6 MV
- phase_space: Generación y manejo de espacios de fase
- dose_calculation: Cálculo de distribuciones de dosis en phantoms

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

__version__ = '1.0.0'

# Imports principales para facilitar el uso
try:
    from .linac_6mv import LinacSimulation
    from .phase_space import PhaseSpaceGenerator, PhaseSpaceSimulation
    from .dose_calculation import DoseCalculator
    
    __all__ = [
        'LinacSimulation',
        'PhaseSpaceGenerator',
        'PhaseSpaceSimulation',
        'DoseCalculator',
    ]
except ImportError as e:
    # Si opengate no está instalado, no fallar al importar el paquete
    print(f"⚠ Advertencia: No se pudieron importar todos los módulos de simulación")
    print(f"  Razón: {e}")
    print(f"  Instala opengate con: pip install opengate")
    __all__ = []
