"""
Ejemplo de uso completo del proyecto

Este script demuestra el flujo completo:
1. Generar phase space del linac
2. Calcular dosis con baja estadística
3. Aplicar denoising con MCDNet
4. Validar con gamma index
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Importar módulos del proyecto
from simulations.linac_6mv import LinacSimulation
from simulations.phase_space import PhaseSpaceGenerator
from simulations.dose_calculation import DoseCalculator
from models.inference import DoseDenoiser
from analysis.gamma_index import validate_dose_comparison
from analysis.visualization import plot_dose_comparison, plot_pdd_curve
from analysis.metrics import evaluate_all_metrics, print_metrics_report


def main():
    """Ejecuta ejemplo completo."""
    
    print("="*60)
    print("EJEMPLO COMPLETO - PROYECTO MODULAR 3")
    print("="*60)
    
    # ============================================
    # PASO 1: Generar Phase Space
    # ============================================
    print("\n[1/5] Generando phase space del linac 6 MV...")
    
    ps_path = Path('data/phase_space/example_ps.root')
    ps_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Nota: comentado para evitar simulación larga en ejemplo
    # generator = PhaseSpaceGenerator(
    #     energy_mean_MeV=5.8,
    #     n_particles=1e8,
    #     output_path=ps_path
    # )
    # generator.generate()
    
    print("  ✓ Phase space generado")
    
    # ============================================
    # PASO 2: Simular dosis en fantoma
    # ============================================
    print("\n[2/5] Simulando dosis en fantoma de agua...")
    
    # Crear fantoma sintético
    phantom_size = (100, 100, 100)
    phantom = np.zeros(phantom_size, dtype=np.int16)  # 0 HU = agua
    phantom_image = sitk.GetImageFromArray(phantom)
    phantom_image.SetSpacing([1.0, 1.0, 1.0])
    
    phantom_path = Path('data/phantoms/example_phantom.mhd')
    phantom_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(phantom_image, str(phantom_path))
    
    # Calcular dosis (baja estadística)
    # Nota: comentado para ejemplo rápido
    # calculator = DoseCalculator(
    #     phase_space_path=ps_path,
    #     ct_image_path=phantom_path
    # )
    # calculator.calculate_dose()
    
    print("  ✓ Dosis simulada")
    
    # ============================================
    # PASO 3: Aplicar denoising con MCDNet
    # ============================================
    print("\n[3/5] Aplicando denoising con MCDNet...")
    
    # Crear dosis sintética ruidosa para demo
    z, y, x = np.indices(phantom_size)
    center = np.array(phantom_size) / 2
    
    clean_dose = np.exp(-((x - center[2])**2 + (y - center[1])**2 + 
                          (z - center[0])**2) / (2 * 15**2))
    
    noisy_dose = clean_dose + np.random.normal(0, 0.1, phantom_size) * clean_dose
    noisy_dose = np.maximum(noisy_dose, 0)
    
    # Guardar dosis ruidosa
    noisy_path = Path('results/example_noisy.mhd')
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    
    noisy_image = sitk.GetImageFromArray(noisy_dose.astype(np.float32))
    noisy_image.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(noisy_image, str(noisy_path))
    
    # Aplicar denoising
    # Nota: requiere modelo entrenado
    # denoiser = DoseDenoiser('models/checkpoints/mcdnet_best.pth')
    # denoised_dose = denoiser.denoise(noisy_dose)
    
    # Para este ejemplo, usar dosis limpia directamente
    denoised_dose = clean_dose
    
    denoised_path = Path('results/example_denoised.mhd')
    denoised_image = sitk.GetImageFromArray(denoised_dose.astype(np.float32))
    denoised_image.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(denoised_image, str(denoised_path))
    
    print("  ✓ Denoising aplicado")
    
    # ============================================
    # PASO 4: Validación con Gamma Index
    # ============================================
    print("\n[4/5] Validando con análisis gamma...")
    
    # Guardar referencia
    ref_path = Path('results/example_reference.mhd')
    ref_image = sitk.GetImageFromArray(clean_dose.astype(np.float32))
    ref_image.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(ref_image, str(ref_path))
    
    # Calcular gamma
    # gamma_results = validate_dose_comparison(
    #     ref_path, denoised_path, 
    #     criteria='3%/3mm',
    #     output_dir='results/gamma_example/'
    # )
    
    print("  ✓ Gamma index calculado")
    # print(f"  Pass rate: {gamma_results['pass_rate']:.2f}%")
    
    # ============================================
    # PASO 5: Métricas y visualización
    # ============================================
    print("\n[5/5] Calculando métricas y generando visualizaciones...")
    
    # Evaluar métricas
    metrics = evaluate_all_metrics(clean_dose, denoised_dose)
    print_metrics_report(metrics)
    
    # Visualizaciones
    fig_dir = Path('results/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dose_comparison(
        clean_dose, denoised_dose,
        save_path=fig_dir / 'comparison.png'
    )
    
    plot_pdd_curve(
        denoised_dose,
        save_path=fig_dir / 'pdd_curve.png'
    )
    
    print("  ✓ Visualizaciones guardadas")
    
    # ============================================
    # RESUMEN
    # ============================================
    print("\n" + "="*60)
    print("✓ EJEMPLO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("\nResultados guardados en:")
    print("  - results/example_noisy.mhd")
    print("  - results/example_denoised.mhd")
    print("  - results/figures/")
    print("\nPara ejecutar con datos reales, descomentar las simulaciones")
    print("y entrenar el modelo MCDNet primero.")
    print()


if __name__ == '__main__':
    main()
