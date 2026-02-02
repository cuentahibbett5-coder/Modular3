#!/usr/bin/env python3
"""
Detailed analysis of dose distributions
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def read_mhd_file(mhd_path):
    """Read MetaImage (.mhd) file and corresponding .raw data"""
    mhd_path = Path(mhd_path)
    
    # Parse MHD header
    metadata = {}
    with open(mhd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    # Extract information
    dim_size = list(map(int, metadata['DimSize'].split()))
    element_spacing = list(map(float, metadata['ElementSpacing'].split()))
    element_type = metadata.get('ElementType', 'MET_FLOAT')
    byte_order_msb = metadata.get('BinaryDataByteOrderMSB', 'False') == 'True'
    offset = list(map(float, metadata['Offset'].split()))
    
    # Read data
    raw_file = mhd_path.parent / metadata['ElementDataFile']
    byte_order = '>' if byte_order_msb else '<'
    if element_type == 'MET_DOUBLE':
        dtype_char = 'd'
    else:
        dtype_char = 'f'
    
    data = np.fromfile(raw_file, dtype=byte_order + dtype_char)
    data = data.reshape(dim_size).astype(np.float32)
    
    return data, element_spacing, offset, metadata


def calculate_statistics(dose_data, element_spacing):
    """Calculate dose statistics"""
    valid_doses = dose_data[dose_data > 0]
    
    stats = {
        'max_dose': dose_data.max(),
        'mean_dose': valid_doses.mean() if len(valid_doses) > 0 else 0,
        'min_dose': valid_doses.min() if len(valid_doses) > 0 else 0,
        'std_dose': valid_doses.std() if len(valid_doses) > 0 else 0,
        'num_voxels': dose_data.size,
        'num_dose_voxels': np.count_nonzero(dose_data),
        'coverage': 100 * np.count_nonzero(dose_data) / dose_data.size,
        'volume_cm3': (dose_data.size * np.prod(element_spacing) * 1e-3),
    }
    
    # Calculate dose-volume histogram values
    sorted_doses = np.sort(valid_doses)[::-1]
    stats['d_95'] = sorted_doses[int(0.95*len(sorted_doses))] if len(sorted_doses) > 0 else 0
    stats['d_50'] = sorted_doses[int(0.50*len(sorted_doses))] if len(sorted_doses) > 0 else 0
    
    return stats


def plot_dose_volume_histogram(dose_data, output_prefix):
    """Plot dose-volume histogram"""
    valid_doses = dose_data[dose_data > 0]
    sorted_doses = np.sort(valid_doses)[::-1]
    
    cumulative_volume = np.arange(len(sorted_doses)) / len(sorted_doses) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_doses, cumulative_volume, 'b-', linewidth=2)
    ax.fill_between(sorted_doses, cumulative_volume, alpha=0.3)
    
    ax.set_xlabel('Dosis (Gy)')
    ax.set_ylabel('Volumen (%)')
    ax.set_title('Histograma Dosis-Volumen (DVH)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, sorted_doses.max() * 1.1])
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = Path(output_prefix).parent / f"{Path(output_prefix).stem}_dvh.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ DVH guardado: {output_path}")
    plt.close()


def plot_dose_histogram(dose_data, output_prefix):
    """Plot dose distribution histogram"""
    valid_doses = dose_data[dose_data > 0].flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_doses, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.axvline(valid_doses.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {valid_doses.mean():.6f} Gy')
    ax.axvline(np.median(valid_doses), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(valid_doses):.6f} Gy')
    
    ax.set_xlabel('Dosis (Gy)')
    ax.set_ylabel('Número de Voxels')
    ax.set_title('Distribución de Dosis')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_prefix).parent / f"{Path(output_prefix).stem}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Histograma guardado: {output_path}")
    plt.close()


def plot_dose_comparison(dose1_file, dose2_file, output_dir):
    """Compare two dose distributions"""
    dose1, spacing1, _, _ = read_mhd_file(dose1_file)
    dose2, spacing2, _, _ = read_mhd_file(dose2_file)
    
    # Ensure same shape
    if dose1.shape != dose2.shape:
        print(f"⚠️ Las dosis tienen diferente tamaño: {dose1.shape} vs {dose2.shape}")
        return
    
    # Calculate difference
    diff = dose1 - dose2
    rel_diff = np.zeros_like(diff)
    mask = dose1 > 0
    rel_diff[mask] = (diff[mask] / dose1[mask]) * 100
    
    # Statistics
    print(f"\n📊 Comparación de Dosis:")
    print(f"   Diferencia absoluta media: {np.mean(np.abs(diff)):.6f} Gy")
    print(f"   Diferencia relativa media: {np.mean(np.abs(rel_diff[mask])):.2f}%")
    print(f"   Máxima diferencia: {np.max(np.abs(diff)):.6f} Gy")
    
    # Plot comparison
    cx, cy, cz = [s // 2 for s in dose1.shape]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparación de Distribuciones de Dosis', fontsize=14, fontweight='bold')
    
    # Dose 1
    im0 = axes[0, 0].imshow(dose1[:, :, cz], cmap='jet')
    axes[0, 0].set_title('Dosis 1 (Axial)')
    plt.colorbar(im0, ax=axes[0, 0], label='Gy')
    
    # Dose 2
    im1 = axes[0, 1].imshow(dose2[:, :, cz], cmap='jet')
    axes[0, 1].set_title('Dosis 2 (Axial)')
    plt.colorbar(im1, ax=axes[0, 1], label='Gy')
    
    # Difference
    im2 = axes[1, 0].imshow(diff[:, :, cz], cmap='RdBu_r')
    axes[1, 0].set_title('Diferencia (Dosis 1 - Dosis 2)')
    plt.colorbar(im2, ax=axes[1, 0], label='Gy')
    
    # Relative difference
    im3 = axes[1, 1].imshow(rel_diff[:, :, cz], cmap='RdBu_r', vmin=-50, vmax=50)
    axes[1, 1].set_title('Diferencia Relativa (%)')
    plt.colorbar(im3, ax=axes[1, 1], label='%')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'dose_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparación guardada: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze dose distributions')
    parser.add_argument('--dose', type=str, required=True, help='Dose MHD file')
    parser.add_argument('--compare', type=str, help='Second dose MHD file for comparison')
    parser.add_argument('--output', type=str, help='Output directory (default: same as dose file dir)')
    
    args = parser.parse_args()
    
    dose_path = Path(args.dose)
    output_dir = Path(args.output) if args.output else dose_path.parent
    output_prefix = str(output_dir / dose_path.stem)
    
    print("\n" + "="*80)
    print("ANÁLISIS DETALLADO DE DOSIS")
    print("="*80)
    print(f"📂 Archivo: {dose_path}")
    
    # Read dose
    dose_data, spacing, offset, metadata = read_mhd_file(dose_path)
    
    # Calculate statistics
    stats = calculate_statistics(dose_data, spacing)
    
    print(f"\n📊 Estadísticas de Dosis:")
    print(f"   Dosis máxima: {stats['max_dose']:.6f} Gy")
    print(f"   Dosis media: {stats['mean_dose']:.6f} Gy")
    print(f"   Dosis mediana: {stats['d_50']:.6f} Gy")
    print(f"   Desviación estándar: {stats['std_dose']:.6f} Gy")
    print(f"   Voxels con dosis: {stats['num_dose_voxels']:,} / {stats['num_voxels']:,}")
    print(f"   Cobertura: {stats['coverage']:.2f}%")
    print(f"   D95: {stats['d_95']:.6f} Gy")
    print(f"   Volumen: {stats['volume_cm3']:.2f} cm³")
    
    # Generate plots
    print(f"\n🎨 Generando gráficos...")
    plot_dose_histogram(dose_data, output_prefix)
    plot_dose_volume_histogram(dose_data, output_prefix)
    
    # Comparison if requested
    if args.compare:
        print(f"\n📊 Comparando con: {args.compare}")
        plot_dose_comparison(dose_path, Path(args.compare), output_dir)
    
    print(f"\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
