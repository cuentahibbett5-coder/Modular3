#!/usr/bin/env python3
"""
Visualize dose distributions from MHD files
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


def read_mhd_file(mhd_path):
    """Read MetaImage (.mhd) file and its corresponding .raw data"""
    mhd_path = Path(mhd_path)
    
    # Parse MHD header
    metadata = {}
    with open(mhd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    # Extract relevant information
    ndims = int(metadata.get('NDims', 3))
    dim_size = list(map(int, metadata['DimSize'].split()))
    element_spacing = list(map(float, metadata['ElementSpacing'].split()))
    element_type = metadata.get('ElementType', 'MET_FLOAT')
    byte_order_msb = metadata.get('BinaryDataByteOrderMSB', 'False') == 'True'
    offset = list(map(float, metadata['Offset'].split()))
    
    # Determine data type
    if element_type == 'MET_DOUBLE':
        dtype = np.float64
    elif element_type == 'MET_FLOAT':
        dtype = np.float32
    else:
        dtype = np.float32
    
    # Read raw data
    raw_file = mhd_path.parent / metadata['ElementDataFile']
    byte_order = '>' if byte_order_msb else '<'
    
    # Determine numpy dtype string
    if element_type == 'MET_DOUBLE':
        numpy_dtype = byte_order + 'd'  # 8-byte double
    elif element_type == 'MET_FLOAT':
        numpy_dtype = byte_order + 'f'  # 4-byte float
    else:
        numpy_dtype = byte_order + 'f'  # default float
    
    data = np.fromfile(raw_file, dtype=numpy_dtype)
    
    # Reshape to 3D
    data = data.reshape(dim_size).astype(np.float32)
    
    return data, element_spacing, offset, metadata


def plot_dose_slices(dose_data, element_spacing, offset, output_prefix):
    """Create visualization with 3 orthogonal slices"""
    print(f"\n📊 Datos de dosis:")
    print(f"   Shape: {dose_data.shape}")
    print(f"   Espaciado: {element_spacing} mm")
    print(f"   Offset: {offset} mm")
    print(f"   Rango dosis: {dose_data.min():.6e} - {dose_data.max():.6e} Gy")
    print(f"   Dosis media: {dose_data.mean():.6e} Gy")
    print(f"   Dosis media: {dose_data.mean():.6f} Gy")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Distribución de Dosis 6MV Elekta Versa', fontsize=14, fontweight='bold')
    
    # Get center slices
    cx, cy, cz = [s // 2 for s in dose_data.shape]
    
    # X-Y slice (axial, at center Z)
    im0 = axes[0].imshow(dose_data[:, :, cz], cmap='jet', origin='lower')
    axes[0].set_title(f'Axial (Z={cz * element_spacing[2]:.1f} mm)')
    axes[0].set_xlabel('X (píxeles)')
    axes[0].set_ylabel('Y (píxeles)')
    plt.colorbar(im0, ax=axes[0], label='Dosis (Gy)')
    
    # X-Z slice (sagital, at center Y)
    im1 = axes[1].imshow(dose_data[:, cy, :], cmap='jet', origin='lower')
    axes[1].set_title(f'Sagital (Y={cy * element_spacing[1]:.1f} mm)')
    axes[1].set_xlabel('X (píxeles)')
    axes[1].set_ylabel('Z (píxeles)')
    plt.colorbar(im1, ax=axes[1], label='Dosis (Gy)')
    
    # Y-Z slice (coronal, at center X)
    im2 = axes[2].imshow(dose_data[cx, :, :], cmap='jet', origin='lower')
    axes[2].set_title(f'Coronal (X={cx * element_spacing[0]:.1f} mm)')
    axes[2].set_xlabel('Y (píxeles)')
    axes[2].set_ylabel('Z (píxeles)')
    plt.colorbar(im2, ax=axes[2], label='Dosis (Gy)')
    
    plt.tight_layout()
    output_path = Path(output_prefix).parent / f"{Path(output_prefix).stem}_slices.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_dose_profiles(dose_data, element_spacing, offset, output_prefix):
    """Create depth-dose and lateral profiles"""
    cx, cy, cz = [s // 2 for s in dose_data.shape]
    
    # Depth dose curve (along Z axis at center X,Y)
    depth_profile = dose_data[cx, cy, :]
    depth_mm = np.arange(len(depth_profile)) * element_spacing[2]
    
    # Lateral profile (along X axis at center Y,Z)
    lateral_profile = dose_data[:, cy, cz]
    lateral_mm = np.arange(len(lateral_profile)) * element_spacing[0]
    
    # Normalize to max dose
    max_depth = depth_profile.max()
    max_lateral = lateral_profile.max()
    
    if max_depth > 0:
        depth_norm = 100 * depth_profile / max_depth
    else:
        depth_norm = depth_profile
    
    if max_lateral > 0:
        lateral_norm = 100 * lateral_profile / max_lateral
    else:
        lateral_norm = lateral_profile
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Perfiles de Dosis', fontsize=14, fontweight='bold')
    
    # Depth-dose
    axes[0].plot(depth_mm, depth_norm, 'b-', linewidth=2)
    axes[0].fill_between(depth_mm, depth_norm, alpha=0.3)
    axes[0].set_xlabel('Profundidad (mm)')
    axes[0].set_ylabel('Dosis Relativa (%)')
    axes[0].set_title('Profundidad-Dosis (PDD)')
    axes[0].grid(True, alpha=0.3)
    if max_depth > 0:
        axes[0].set_ylim([0, 110])
    
    # Lateral profile
    axes[1].plot(lateral_mm, lateral_norm, 'r-', linewidth=2)
    axes[1].fill_between(lateral_mm, lateral_norm, alpha=0.3, color='red')
    axes[1].set_xlabel('Posición Lateral (mm)')
    axes[1].set_ylabel('Dosis Relativa (%)')
    axes[1].set_title('Perfil Lateral (a profundidad = dosis máxima)')
    axes[1].grid(True, alpha=0.3)
    if max_lateral > 0:
        axes[1].set_ylim([0, 110])
    
    plt.tight_layout()
    output_path = Path(output_prefix).parent / f"{Path(output_prefix).stem}_profiles.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_isodose_contours(dose_data, element_spacing, output_prefix):
    """Create isodose contour plot"""
    cx, cy, cz = [s // 2 for s in dose_data.shape]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Curvas de Isodosis', fontsize=14, fontweight='bold')
    
    # Axial view with contours
    axial_slice = dose_data[:, :, cz]
    x_extent = [0, axial_slice.shape[0] * element_spacing[0]]
    y_extent = [0, axial_slice.shape[1] * element_spacing[1]]
    
    im0 = axes[0].imshow(axial_slice, cmap='jet', extent=x_extent + y_extent, origin='lower')
    contours0 = axes[0].contour(axial_slice, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    axes[0].clabel(contours0, inline=True, fontsize=8)
    axes[0].set_title(f'Axial (Z={cz * element_spacing[2]:.1f} mm)')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    plt.colorbar(im0, ax=axes[0], label='Dosis (Gy)')
    
    # Sagital view with contours
    sagital_slice = dose_data[:, cy, :]
    x_extent = [0, sagital_slice.shape[0] * element_spacing[0]]
    z_extent = [0, sagital_slice.shape[1] * element_spacing[2]]
    
    im1 = axes[1].imshow(sagital_slice, cmap='jet', extent=x_extent + z_extent, origin='lower')
    contours1 = axes[1].contour(sagital_slice, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    axes[1].clabel(contours1, inline=True, fontsize=8)
    axes[1].set_title(f'Sagital (Y={cy * element_spacing[1]:.1f} mm)')
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Z (mm)')
    plt.colorbar(im1, ax=axes[1], label='Dosis (Gy)')
    
    plt.tight_layout()
    output_path = Path(output_prefix).parent / f"{Path(output_prefix).stem}_isodose.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize dose distributions')
    parser.add_argument('--dose', type=str, required=True, help='Path to dose MHD file')
    parser.add_argument('--output', type=str, help='Output prefix (default: same as dose file)')
    
    args = parser.parse_args()
    
    dose_path = Path(args.dose)
    if not dose_path.exists():
        print(f"❌ Archivo no encontrado: {dose_path}")
        return 1
    
    output_prefix = args.output or str(dose_path)
    
    print("\n" + "="*80)
    print("VISUALIZACIÓN DE DOSIS")
    print("="*80)
    print(f"📂 Archivo: {dose_path}")
    
    # Read dose data
    dose_data, spacing, offset, metadata = read_mhd_file(dose_path)
    
    # Create visualizations
    plot_dose_slices(dose_data, spacing, offset, output_prefix)
    plot_dose_profiles(dose_data, spacing, offset, output_prefix)
    plot_isodose_contours(dose_data, spacing, output_prefix)
    
    print("\n" + "="*80)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
