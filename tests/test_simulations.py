"""
Tests unitarios para módulos de simulación

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

import opengate as gate


class TestLinacSimulation(unittest.TestCase):
    """Tests para LinacSimulation."""
    
    def test_simulation_initialization(self):
        """Verifica inicialización de simulación."""
        from simulations.linac_6mv import LinacSimulation
        
        linac = LinacSimulation(
            energy_mean_MeV=5.8,
            energy_sigma_percent=3.0,
            spot_size_mm=3.0
        )
        
        self.assertIsNotNone(linac.sim)
        self.assertEqual(linac.energy_mean_MeV, 5.8)
    
    def test_geometry_creation(self):
        """Verifica creación de geometría."""
        from simulations.linac_6mv import LinacSimulation
        
        linac = LinacSimulation()
        linac.setup_geometry()
        
        # Verificar que componentes existen
        self.assertIn('target', linac.sim.volume_manager.volumes)
        self.assertIn('flattening_filter', linac.sim.volume_manager.volumes)


class TestPhaseSpace(unittest.TestCase):
    """Tests para generación de phase space."""
    
    def test_phase_space_generator(self):
        """Verifica generador de phase space."""
        from simulations.phase_space import PhaseSpaceGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_phase_space.root'
            
            generator = PhaseSpaceGenerator(
                energy_mean_MeV=5.8,
                n_particles=1000,  # Pocos para test rápido
                output_path=output_path
            )
            
            # Nota: no ejecutar simulación completa en test
            # generator.generate()
            
            self.assertEqual(generator.n_particles, 1000)


class TestDoseCalculation(unittest.TestCase):
    """Tests para cálculo de dosis."""
    
    def test_dose_calculator_init(self):
        """Verifica inicialización de calculador de dosis."""
        from simulations.dose_calculation import DoseCalculator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ps_path = Path(tmpdir) / 'phase_space.root'
            ct_path = Path(tmpdir) / 'ct.mhd'
            
            # Crear archivos dummy
            ps_path.touch()
            ct_path.touch()
            
            calc = DoseCalculator(
                phase_space_path=ps_path,
                ct_image_path=ct_path
            )
            
            self.assertIsNotNone(calc)


class TestMCDNet(unittest.TestCase):
    """Tests para arquitectura MCDNet."""
    
    def test_model_creation(self):
        """Verifica creación de modelo."""
        from models.mcdnet import create_mcdnet
        import torch
        
        model = create_mcdnet('standard')
        self.assertIsNotNone(model)
        
        # Verificar forward pass
        x = torch.randn(1, 1, 80, 80, 80)
        y = model(x)
        
        self.assertEqual(y.shape, x.shape)
    
    def test_model_output_range(self):
        """Verifica que output esté en rango razonable."""
        from models.mcdnet import MCDNet3D
        import torch
        
        model = MCDNet3D()
        model.eval()
        
        x = torch.randn(1, 1, 64, 64, 64) * 0.5 + 0.5  # [0, 1]
        
        with torch.no_grad():
            y = model(x)
        
        # Output debe estar en rango similar
        self.assertTrue(torch.all(y >= 0))
        self.assertTrue(torch.all(y <= 2))


class TestGammaIndex(unittest.TestCase):
    """Tests para análisis gamma."""
    
    def test_gamma_calculation_identical(self):
        """Verifica gamma = 0 para dosis idénticas."""
        from analysis.gamma_index import calculate_gamma_index
        
        # Crear dosis sintéticas idénticas
        dose = np.random.rand(50, 50, 50) * 100
        
        pass_rate, gamma_map = calculate_gamma_index(
            dose, dose, 
            dose_percent_threshold=3,
            distance_mm_threshold=3
        )
        
        # Pass rate debe ser 100% para dosis idénticas
        self.assertAlmostEqual(pass_rate, 100.0, delta=1.0)
    
    def test_gamma_calculation_different(self):
        """Verifica gamma > 0 para dosis diferentes."""
        from analysis.gamma_index import calculate_gamma_index
        
        # Crear dosis diferentes
        dose1 = np.random.rand(50, 50, 50) * 100
        dose2 = dose1 * 1.1  # 10% diferencia
        
        pass_rate, gamma_map = calculate_gamma_index(
            dose1, dose2,
            dose_percent_threshold=3,
            distance_mm_threshold=3
        )
        
        # Pass rate debe ser < 100% para dosis diferentes
        self.assertLess(pass_rate, 100.0)


class TestMetrics(unittest.TestCase):
    """Tests para métricas de evaluación."""
    
    def test_mse_calculation(self):
        """Verifica cálculo de MSE."""
        from analysis.metrics import calculate_mse
        
        ref = np.array([1, 2, 3, 4, 5])
        eval = np.array([1, 2, 3, 4, 5])
        
        mse = calculate_mse(ref, eval)
        self.assertAlmostEqual(mse, 0.0)
    
    def test_psnr_calculation(self):
        """Verifica cálculo de PSNR."""
        from analysis.metrics import calculate_psnr
        
        ref = np.random.rand(10, 10, 10)
        eval = ref.copy()
        
        psnr = calculate_psnr(ref, eval)
        
        # PSNR debe ser infinito para señales idénticas
        self.assertEqual(psnr, float('inf'))


def run_tests():
    """Ejecuta todos los tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
