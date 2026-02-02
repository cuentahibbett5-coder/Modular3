"""
Generador de dataset para entrenamiento de MCDNet

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse
from tqdm import tqdm

import opengate as gate
from simulations.dose_calculation import DoseCalculator


class DatasetGenerator:
    """Genera pares (dosis baja, dosis alta) para entrenamiento."""
    
    def __init__(self, output_dir, phantom_types=['water', 'bone', 'lung'], 
                 field_sizes=[5, 10, 15]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.phantom_types = phantom_types
        self.field_sizes = field_sizes
        
        self.low_particles = 1e7
        self.high_particles = 1e9
    
    def create_phantom(self, phantom_type, size=(30, 30, 30)):
        """
        Crea fantoma homogéneo.
        
        Args:
            phantom_type: 'water', 'bone', 'lung'
            size: Tamaño en cm
        
        Returns:
            Imagen SimpleITK con densidades HU
        """
        # Definir valores HU
        hu_values = {
            'water': 0,
            'bone': 1000,
            'lung': -600,
            'adipose': -100
        }
        
        hu = hu_values.get(phantom_type, 0)
        
        # Crear array con voxels de 1mm
        voxels = [int(s * 10) for s in size]  # cm to mm
        phantom = np.full(voxels, hu, dtype=np.int16)
        
        # Convertir a imagen
        image = sitk.GetImageFromArray(phantom)
        image.SetSpacing([1.0, 1.0, 1.0])  # mm
        image.SetOrigin([0, 0, 0])
        
        return image
    
    def simulate_dose(self, phantom_image, field_size, n_particles, output_path):
        """
        Simula dosis en fantoma.
        
        Args:
            phantom_image: Imagen CT del fantoma
            field_size: Tamaño de campo (cm)
            n_particles: Número de partículas
            output_path: Ruta de salida
        """
        # Configurar simulación
        calc = DoseCalculator(
            phase_space_path=f'data/phase_space/linac_6mv_10x10.root',
            ct_image_path=None  # Usar phantom_image directamente
        )
        
        # Crear simulación
        sim = gate.Simulation()
        
        # Aquí iría la configuración completa...
        # (simplificado para este ejemplo)
        
        print(f"  Simulando {n_particles:.0e} partículas...")
        # sim.run()
        
        # Guardar dosis
        # dose_image = ...
        # sitk.WriteImage(dose_image, str(output_path))
    
    def generate_sample(self, sample_id, phantom_type, field_size):
        """Genera un par de muestras (low, high)."""
        # Crear fantoma
        phantom = self.create_phantom(phantom_type)
        
        # Rutas de salida
        low_path = self.output_dir / f'sample_{sample_id:03d}_low_dose.mhd'
        high_path = self.output_dir / f'sample_{sample_id:03d}_clean_dose.mhd'
        phantom_path = self.output_dir / f'sample_{sample_id:03d}_phantom.mhd'
        
        # Guardar fantoma
        sitk.WriteImage(phantom, str(phantom_path))
        
        # Simular dosis baja estadística
        print(f"Sample {sample_id}: {phantom_type}, {field_size}x{field_size} cm²")
        # self.simulate_dose(phantom, field_size, self.low_particles, low_path)
        
        # Simular dosis alta estadística
        # self.simulate_dose(phantom, field_size, self.high_particles, high_path)
        
        # Por ahora, crear datos sintéticos de ejemplo
        self._create_synthetic_pair(low_path, high_path)
    
    def _create_synthetic_pair(self, low_path, high_path):
        """Crea par sintético para pruebas (TEMPORAL)."""
        # Generar dosis limpia sintética
        size = (100, 100, 100)
        z, y, x = np.indices(size)
        
        # Distribución gaussiana 3D simulando haz
        center = np.array(size) / 2
        sigma = 15
        
        clean_dose = np.exp(-((x - center[2])**2 + (y - center[1])**2 + 
                              (z - center[0])**2) / (2 * sigma**2))
        
        # Añadir ruido para versión "low statistics"
        noise_level = 0.15
        noisy_dose = clean_dose + np.random.normal(0, noise_level, size) * clean_dose
        noisy_dose = np.maximum(noisy_dose, 0)
        
        # Guardar
        clean_image = sitk.GetImageFromArray(clean_dose.astype(np.float32))
        noisy_image = sitk.GetImageFromArray(noisy_dose.astype(np.float32))
        
        clean_image.SetSpacing([1.0, 1.0, 1.0])
        noisy_image.SetSpacing([1.0, 1.0, 1.0])
        
        sitk.WriteImage(clean_image, str(high_path))
        sitk.WriteImage(noisy_image, str(low_path))
    
    def generate_dataset(self, n_samples_per_config=10):
        """
        Genera dataset completo.
        
        Args:
            n_samples_per_config: Muestras por configuración (phantom, field)
        """
        sample_id = 0
        
        for phantom_type in self.phantom_types:
            for field_size in self.field_sizes:
                for i in range(n_samples_per_config):
                    self.generate_sample(sample_id, phantom_type, field_size)
                    sample_id += 1
        
        print(f"\n✓ Dataset generado: {sample_id} pares de muestras")
        print(f"  Ubicación: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generar dataset de entrenamiento')
    parser.add_argument('--output', type=str, default='data/training',
                       help='Directorio de salida')
    parser.add_argument('--phantoms', nargs='+', default=['water', 'bone', 'lung'],
                       help='Tipos de fantoma')
    parser.add_argument('--fields', nargs='+', type=int, default=[5, 10, 15],
                       help='Tamaños de campo (cm)')
    parser.add_argument('--samples-per-config', type=int, default=10,
                       help='Muestras por configuración')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        output_dir=args.output,
        phantom_types=args.phantoms,
        field_sizes=args.fields
    )
    
    generator.generate_dataset(n_samples_per_config=args.samples_per_config)


if __name__ == '__main__':
    main()
