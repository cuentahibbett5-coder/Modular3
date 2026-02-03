#!/usr/bin/env python3
"""
Visualiza datasets de dosis IAEA con diferentes estadísticas.
Genera visualizaciones 2D (slices) y 3D (Plotly interactivo).
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import plotly.graph_objects as go


def read_dose(dose_file):
    """Lee archivo MHD de dosis."""
    img = sitk.ReadImage(str(dose_file))
    dose = sitk.GetArrayFromImage(img)  # Z, Y, X
    spacing = img.GetSpacing()  # X, Y, Z
    return dose, spacing


def get_dataset_stats(dose, label):
    """Calcula estadísticas de la dosis."""
    max_val = dose.max()
    mean_val = dose[dose > 0].mean() if (dose > 0).any() else 0
    nonzero = (dose > 0).sum()
    total = dose.size
    
    return {
        'label': label,
        'max': max_val,
        'mean': mean_val,
        'nonzero': nonzero,
        'nonzero_frac': 100 * nonzero / total
    }


def visualize_2d_comparison(datasets_dir, output_file=None):
    """Genera comparación 2D de todos los datasets."""
    
    datasets_dir = Path(datasets_dir)
    dose_dirs = sorted([d for d in datasets_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('dose_')])
    
    if not dose_dirs:
        print("No se encontraron datasets")
        return
    
    n_datasets = len(dose_dirs)
    
    # Cargar todos los datasets
    doses = []
    labels = []
    stats_list = []
    
    for dose_dir in dose_dirs:
        label = dose_dir.name.replace('dose_', '')
        dose_file = dose_dir / 'dose_dose.mhd'
        
        if not dose_file.exists():
            continue
        
        dose, spacing = read_dose(dose_file)
        doses.append(dose)
        labels.append(label)
        stats_list.append(get_dataset_stats(dose, label))
    
    # Normalizar todos a la misma escala (por el máximo del dataset más grande)
    global_max = max(d.max() for d in doses)
    
    # Crear figura
    fig, axes = plt.subplots(3, n_datasets, figsize=(4*n_datasets, 12))
    
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (dose, label, stats) in enumerate(zip(doses, labels, stats_list)):
        # Índices centrales
        cz, cy, cx = [s // 2 for s in dose.shape]
        
        # Axial (Z central)
        ax = axes[0, i]
        im = ax.imshow(dose[cz], cmap='hot', vmin=0, vmax=global_max)
        ax.set_title(f'{label}\nAxial (Z={cz})')
        ax.axis('off')
        
        # Sagital (X central)
        ax = axes[1, i]
        ax.imshow(dose[:, :, cx], cmap='hot', vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(f'Sagital (X={cx})')
        ax.axis('off')
        
        # Coronal (Y central)
        ax = axes[2, i]
        ax.imshow(dose[:, cy, :], cmap='hot', vmin=0, vmax=global_max, aspect='auto')
        ax.set_title(f'Coronal (Y={cy})')
        ax.axis('off')
    
    plt.suptitle('Comparación de Datasets IAEA - Electrones 6 MeV', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Guardado: {output_file}")
    
    plt.close()
    
    # Imprimir estadísticas
    print("\n📊 Estadísticas de los datasets:")
    print("-" * 60)
    for s in stats_list:
        print(f"  {s['label']:>10}: max={s['max']:.2e}, voxels>0={s['nonzero_frac']:.1f}%")


def visualize_3d_volume(dose, spacing, output_html, title, step=2, threshold=0.05):
    """Genera visualización 3D interactiva con Plotly."""
    
    # Downsample
    dose_ds = dose[::step, ::step, ::step]
    
    # Normalizar
    max_val = dose_ds.max()
    if max_val <= 0:
        max_val = 1.0
    dose_norm = dose_ds / max_val
    
    # Coordenadas
    nz, ny, nx = dose_ds.shape
    z = np.arange(nz) * spacing[2] * step
    y = np.arange(ny) * spacing[1] * step
    x = np.arange(nx) * spacing[0] * step
    
    # Crear meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Solo mostrar voxeles por encima del umbral
    mask = dose_norm > threshold
    
    fig = go.Figure(data=go.Scatter3d(
        x=X[mask.T].flatten(),
        y=Y[mask.T].flatten(),
        z=Z[mask.T].flatten(),
        mode='markers',
        marker=dict(
            size=3,
            color=dose_norm.T[mask.T].flatten(),
            colorscale='Hot',
            opacity=0.8,
            colorbar=dict(title='Dosis (norm)')
        )
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(str(output_html), include_plotlyjs='cdn')


def visualize_3d_all(datasets_dir, output_dir=None, step=3):
    """Genera visualizaciones 3D para todos los datasets."""
    
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir) if output_dir else datasets_dir / '3d'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dose_dirs = sorted([d for d in datasets_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('dose_')])
    
    for dose_dir in dose_dirs:
        label = dose_dir.name.replace('dose_', '')
        dose_file = dose_dir / 'dose_dose.mhd'
        
        if not dose_file.exists():
            continue
        
        print(f"Generando 3D: {label}...")
        dose, spacing = read_dose(dose_file)
        
        output_html = output_dir / f'{label}.html'
        visualize_3d_volume(
            dose, spacing, output_html,
            title=f'Dosis 3D - {label} - Electrones 6 MeV',
            step=step
        )
    
    print(f"✅ HTML guardado en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualiza datasets IAEA')
    parser.add_argument('--datasets-dir', required=True, help='Directorio con datasets')
    parser.add_argument('--output-2d', default=None, help='Archivo PNG para 2D')
    parser.add_argument('--output-3d', default=None, help='Directorio para HTML 3D')
    parser.add_argument('--step', type=int, default=3, help='Downsample step para 3D')
    parser.add_argument('--only-2d', action='store_true', help='Solo generar 2D')
    parser.add_argument('--only-3d', action='store_true', help='Solo generar 3D')
    
    args = parser.parse_args()
    
    datasets_dir = Path(args.datasets_dir)
    
    if not args.only_3d:
        output_2d = args.output_2d or (datasets_dir / 'comparison_2d.png')
        visualize_2d_comparison(datasets_dir, output_2d)
    
    if not args.only_2d:
        output_3d = args.output_3d or (datasets_dir / '3d')
        visualize_3d_all(datasets_dir, output_3d, step=args.step)


if __name__ == '__main__':
    main()
