"""
Análisis Gamma Index para validación de dosis

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from pymedphys import gamma


def calculate_gamma_index(reference_dose, evaluated_dose, 
                          dose_percent_threshold=3, distance_mm_threshold=3,
                          dose_threshold=10):
    """
    Calcula el índice gamma entre dos distribuciones de dosis.
    
    Args:
        reference_dose: Dosis de referencia (ground truth)
        evaluated_dose: Dosis evaluada (simulación/denoised)
        dose_percent_threshold: Tolerancia de dosis (%)
        distance_mm_threshold: Tolerancia de distancia (mm)
        dose_threshold: Umbral mínimo de dosis (% del máximo)
    
    Returns:
        gamma_pass_rate: Porcentaje de puntos con gamma <= 1
        gamma_map: Mapa de valores gamma
    """
    # Normalizar dosis al máximo de referencia
    ref_max = np.max(reference_dose)
    ref_norm = reference_dose / ref_max * 100
    eval_norm = evaluated_dose / ref_max * 100
    
    # Crear coordenadas (asumiendo 1mm de espaciado)
    coords_z = np.arange(ref_norm.shape[0])
    coords_y = np.arange(ref_norm.shape[1])
    coords_x = np.arange(ref_norm.shape[2])
    
    # Calcular gamma
    gamma_map = gamma(
        coords_z, coords_y, coords_x,
        ref_norm,
        coords_z, coords_y, coords_x,
        eval_norm,
        dose_percent_threshold,
        distance_mm_threshold,
        lower_percent_dose_cutoff=dose_threshold,
        interp_fraction=10,
        max_gamma=2.0,
        local_gamma=False,
        quiet=False
    )
    
    # Calcular pass rate
    valid_points = gamma_map[~np.isnan(gamma_map)]
    gamma_pass_rate = np.sum(valid_points <= 1.0) / len(valid_points) * 100
    
    return gamma_pass_rate, gamma_map


def analyze_gamma_statistics(gamma_map):
    """Calcula estadísticas del mapa gamma."""
    valid_gamma = gamma_map[~np.isnan(gamma_map)]
    
    stats = {
        'mean': np.mean(valid_gamma),
        'std': np.std(valid_gamma),
        'min': np.min(valid_gamma),
        'max': np.max(valid_gamma),
        'median': np.median(valid_gamma),
        'pass_rate_1.0': np.sum(valid_gamma <= 1.0) / len(valid_gamma) * 100,
        'pass_rate_1.1': np.sum(valid_gamma <= 1.1) / len(valid_gamma) * 100,
    }
    
    return stats


def plot_gamma_results(gamma_map, slice_idx=None, save_path=None):
    """Visualiza resultados de análisis gamma."""
    if slice_idx is None:
        slice_idx = gamma_map.shape[0] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa gamma en slice
    im1 = axes[0].imshow(gamma_map[slice_idx, :, :], cmap='jet', vmin=0, vmax=2)
    axes[0].set_title(f'Gamma Map (Slice {slice_idx})')
    axes[0].set_xlabel('X [voxels]')
    axes[0].set_ylabel('Y [voxels]')
    plt.colorbar(im1, ax=axes[0], label='Gamma Index')
    
    # Histograma
    valid_gamma = gamma_map[~np.isnan(gamma_map)]
    axes[1].hist(valid_gamma, bins=50, range=(0, 2), alpha=0.7, edgecolor='black')
    axes[1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Gamma = 1.0')
    axes[1].set_xlabel('Gamma Index')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Gamma Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico guardado: {save_path}")
    else:
        plt.show()


def validate_dose_comparison(ref_path, eval_path, criteria='3%/3mm', output_dir=None):
    """
    Valida comparación completa de dosis con criterio gamma.
    
    Args:
        ref_path: Ruta a dosis de referencia
        eval_path: Ruta a dosis evaluada
        criteria: '3%/3mm' o '2%/2mm'
        output_dir: Directorio para guardar resultados
    
    Returns:
        dict con resultados de validación
    """
    # Cargar dosis
    ref_image = sitk.ReadImage(str(ref_path))
    eval_image = sitk.ReadImage(str(eval_path))
    
    ref_dose = sitk.GetArrayFromImage(ref_image)
    eval_dose = sitk.GetArrayFromImage(eval_image)
    
    # Configurar criterio
    if criteria == '3%/3mm':
        dose_thresh, dist_thresh = 3, 3
    elif criteria == '2%/2mm':
        dose_thresh, dist_thresh = 2, 2
    else:
        raise ValueError(f"Criterio desconocido: {criteria}")
    
    # Calcular gamma
    print(f"\nCalculando Gamma Index con criterio {criteria}...")
    pass_rate, gamma_map = calculate_gamma_index(
        ref_dose, eval_dose, dose_thresh, dist_thresh
    )
    
    # Estadísticas
    stats = analyze_gamma_statistics(gamma_map)
    
    print(f"\n=== Resultados Gamma {criteria} ===")
    print(f"Pass Rate (γ ≤ 1.0): {pass_rate:.2f}%")
    print(f"Mean Gamma: {stats['mean']:.3f}")
    print(f"Median Gamma: {stats['median']:.3f}")
    print(f"Max Gamma: {stats['max']:.3f}")
    
    # Guardar resultados
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar mapa gamma
        gamma_image = sitk.GetImageFromArray(gamma_map.astype(np.float32))
        gamma_image.CopyInformation(ref_image)
        sitk.WriteImage(gamma_image, str(output_dir / f'gamma_map_{criteria}.mhd'))
        
        # Guardar visualización
        plot_gamma_results(gamma_map, save_path=output_dir / f'gamma_plot_{criteria}.png')
    
    return {
        'pass_rate': pass_rate,
        'stats': stats,
        'gamma_map': gamma_map,
        'criteria': criteria
    }


def main():
    """Script principal de validación gamma."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis Gamma Index')
    parser.add_argument('--reference', type=str, required=True, help='Dosis de referencia')
    parser.add_argument('--evaluated', type=str, required=True, help='Dosis evaluada')
    parser.add_argument('--criteria', type=str, default='3%/3mm', choices=['3%/3mm', '2%/2mm'])
    parser.add_argument('--output', type=str, help='Directorio de salida')
    
    args = parser.parse_args()
    
    results = validate_dose_comparison(
        args.reference, args.evaluated, args.criteria, args.output
    )
    
    # Evaluar criterio de aceptación
    if results['pass_rate'] >= 95.0:
        print(f"\n✓ APROBADO: Pass rate {results['pass_rate']:.2f}% ≥ 95%")
    else:
        print(f"\n✗ RECHAZADO: Pass rate {results['pass_rate']:.2f}% < 95%")


if __name__ == '__main__':
    main()
