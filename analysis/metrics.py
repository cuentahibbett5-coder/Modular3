"""
Métricas de evaluación para validación de dosis

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import numpy as np
import SimpleITK as sitk
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_mse(reference, evaluated):
    """Calcula Mean Squared Error."""
    return mean_squared_error(reference.flatten(), evaluated.flatten())


def calculate_mae(reference, evaluated):
    """Calcula Mean Absolute Error."""
    return mean_absolute_error(reference.flatten(), evaluated.flatten())


def calculate_rmse(reference, evaluated):
    """Calcula Root Mean Squared Error."""
    return np.sqrt(calculate_mse(reference, evaluated))


def calculate_psnr(reference, evaluated, max_value=None):
    """
    Calcula Peak Signal-to-Noise Ratio.
    
    PSNR = 20 * log10(MAX / RMSE)
    """
    if max_value is None:
        max_value = np.max(reference)
    
    mse = calculate_mse(reference, evaluated)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def calculate_ssim(reference, evaluated):
    """
    Calcula Structural Similarity Index (SSIM) simplificado.
    Para implementación completa usar skimage.metrics.structural_similarity
    """
    from skimage.metrics import structural_similarity
    
    # Normalizar a [0, 1]
    ref_norm = reference / np.max(reference)
    eval_norm = evaluated / np.max(reference)
    
    ssim_value = structural_similarity(
        ref_norm, eval_norm,
        data_range=1.0
    )
    
    return ssim_value


def calculate_correlation(reference, evaluated):
    """Calcula coeficiente de correlación de Pearson."""
    corr, _ = pearsonr(reference.flatten(), evaluated.flatten())
    return corr


def calculate_dose_difference_histogram(reference, evaluated, bins=50):
    """
    Calcula histograma de diferencias de dosis.
    
    Returns:
        bins, histogram, statistics
    """
    diff = evaluated - reference
    diff_percent = (diff / np.max(reference)) * 100
    
    hist, bin_edges = np.histogram(diff_percent, bins=bins, range=(-10, 10))
    
    stats = {
        'mean_diff': np.mean(diff_percent),
        'std_diff': np.std(diff_percent),
        'max_diff': np.max(np.abs(diff_percent)),
        'within_1percent': np.sum(np.abs(diff_percent) <= 1) / diff_percent.size * 100,
        'within_2percent': np.sum(np.abs(diff_percent) <= 2) / diff_percent.size * 100,
        'within_3percent': np.sum(np.abs(diff_percent) <= 3) / diff_percent.size * 100,
    }
    
    return bin_edges[:-1], hist, stats


def evaluate_all_metrics(reference, evaluated):
    """
    Calcula todas las métricas de evaluación.
    
    Returns:
        dict con todas las métricas
    """
    metrics = {
        'MSE': calculate_mse(reference, evaluated),
        'MAE': calculate_mae(reference, evaluated),
        'RMSE': calculate_rmse(reference, evaluated),
        'PSNR': calculate_psnr(reference, evaluated),
        'Correlation': calculate_correlation(reference, evaluated),
    }
    
    # SSIM requiere scikit-image
    try:
        metrics['SSIM'] = calculate_ssim(reference, evaluated)
    except ImportError:
        metrics['SSIM'] = None
    
    # Diferencias de dosis
    _, _, diff_stats = calculate_dose_difference_histogram(reference, evaluated)
    metrics.update({
        'Mean_Diff_%': diff_stats['mean_diff'],
        'Std_Diff_%': diff_stats['std_diff'],
        'Max_Diff_%': diff_stats['max_diff'],
        'Within_1%': diff_stats['within_1percent'],
        'Within_2%': diff_stats['within_2percent'],
        'Within_3%': diff_stats['within_3percent'],
    })
    
    return metrics


def print_metrics_report(metrics):
    """Imprime reporte de métricas."""
    print("\n" + "="*50)
    print("REPORTE DE MÉTRICAS")
    print("="*50)
    
    print("\nErrores Absolutos:")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    
    print("\nMétricas de Calidad:")
    print(f"  PSNR:        {metrics['PSNR']:.2f} dB")
    print(f"  Correlation: {metrics['Correlation']:.4f}")
    if metrics['SSIM'] is not None:
        print(f"  SSIM:        {metrics['SSIM']:.4f}")
    
    print("\nDiferencias de Dosis:")
    print(f"  Mean Diff:   {metrics['Mean_Diff_%']:.3f}%")
    print(f"  Std Diff:    {metrics['Std_Diff_%']:.3f}%")
    print(f"  Max Diff:    {metrics['Max_Diff_%']:.3f}%")
    
    print("\nAgreement:")
    print(f"  Within 1%: {metrics['Within_1%']:.2f}%")
    print(f"  Within 2%: {metrics['Within_2%']:.2f}%")
    print(f"  Within 3%: {metrics['Within_3%']:.2f}%")
    
    print("="*50 + "\n")


def main():
    """Script de evaluación de métricas."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluación de métricas')
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--evaluated', type=str, required=True)
    parser.add_argument('--output', type=str, help='Guardar reporte JSON')
    
    args = parser.parse_args()
    
    # Cargar dosis
    ref_image = sitk.ReadImage(args.reference)
    eval_image = sitk.ReadImage(args.evaluated)
    
    reference = sitk.GetArrayFromImage(ref_image)
    evaluated = sitk.GetArrayFromImage(eval_image)
    
    # Calcular métricas
    metrics = evaluate_all_metrics(reference, evaluated)
    
    # Imprimir reporte
    print_metrics_report(metrics)
    
    # Guardar JSON
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Métricas guardadas en: {args.output}")


if __name__ == '__main__':
    main()
