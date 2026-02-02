"""
Herramientas de visualización para análisis de dosis

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path


def plot_dose_comparison(reference, evaluated, slice_idx=None, save_path=None):
    """Compara dos distribuciones de dosis visualmente."""
    if slice_idx is None:
        slice_idx = reference.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin = 0
    vmax = max(np.max(reference), np.max(evaluated))
    
    # Referencia
    im1 = axes[0].imshow(reference[slice_idx, :, :], cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('Referencia (Alta Estadística)')
    axes[0].set_xlabel('X [voxels]')
    axes[0].set_ylabel('Y [voxels]')
    plt.colorbar(im1, ax=axes[0], label='Dosis [Gy]')
    
    # Evaluada
    im2 = axes[1].imshow(evaluated[slice_idx, :, :], cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title('Evaluada (Denoised)')
    axes[1].set_xlabel('X [voxels]')
    plt.colorbar(im2, ax=axes[1], label='Dosis [Gy]')
    
    # Diferencia
    diff = evaluated[slice_idx, :, :] - reference[slice_idx, :, :]
    im3 = axes[2].imshow(diff, cmap='seismic', vmin=-vmax*0.1, vmax=vmax*0.1)
    axes[2].set_title('Diferencia (Eval - Ref)')
    axes[2].set_xlabel('X [voxels]')
    plt.colorbar(im3, ax=axes[2], label='Δ Dosis [Gy]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico guardado: {save_path}")
    else:
        plt.show()


def plot_pdd_curve(dose_map, depth_axis=0, save_path=None):
    """
    Grafica curva PDD (Percentage Depth Dose).
    
    Args:
        dose_map: Mapa de dosis 3D
        depth_axis: Eje de profundidad (0=Z, 1=Y, 2=X)
        save_path: Ruta para guardar gráfico
    """
    # Extraer PDD en eje central
    if depth_axis == 0:
        pdd_profile = dose_map[:, dose_map.shape[1]//2, dose_map.shape[2]//2]
    elif depth_axis == 1:
        pdd_profile = dose_map[dose_map.shape[0]//2, :, dose_map.shape[2]//2]
    else:
        pdd_profile = dose_map[dose_map.shape[0]//2, dose_map.shape[1]//2, :]
    
    # Normalizar a porcentaje
    pdd_percent = (pdd_profile / np.max(pdd_profile)) * 100
    
    # Profundidad en mm (asumiendo 1mm/voxel)
    depth_mm = np.arange(len(pdd_percent))
    
    plt.figure(figsize=(10, 6))
    plt.plot(depth_mm, pdd_percent, 'b-', linewidth=2, label='PDD')
    plt.axhline(100, color='r', linestyle='--', alpha=0.5, label='Dmax (100%)')
    plt.axhline(80, color='orange', linestyle='--', alpha=0.5, label='R80')
    plt.axhline(50, color='green', linestyle='--', alpha=0.5, label='R50')
    
    plt.xlabel('Profundidad [mm]')
    plt.ylabel('Dosis [%]')
    plt.title('Curva PDD (Percentage Depth Dose)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"PDD guardada: {save_path}")
    else:
        plt.show()


def plot_beam_profile(dose_map, depth_idx, axis='x', save_path=None):
    """
    Grafica perfil transversal del haz.
    
    Args:
        dose_map: Mapa de dosis 3D
        depth_idx: Índice de profundidad
        axis: 'x' o 'y' para perfil horizontal/vertical
        save_path: Ruta para guardar gráfico
    """
    if axis == 'x':
        profile = dose_map[depth_idx, dose_map.shape[1]//2, :]
    else:
        profile = dose_map[depth_idx, :, dose_map.shape[2]//2]
    
    # Normalizar
    profile_norm = (profile / np.max(profile)) * 100
    position_mm = np.arange(len(profile_norm)) - len(profile_norm)//2
    
    plt.figure(figsize=(10, 6))
    plt.plot(position_mm, profile_norm, 'b-', linewidth=2)
    plt.axhline(50, color='r', linestyle='--', alpha=0.5, label='50% (Penumbra)')
    plt.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80%')
    
    plt.xlabel(f'Posición {axis.upper()} [mm]')
    plt.ylabel('Dosis Relativa [%]')
    plt.title(f'Perfil del Haz - Profundidad {depth_idx} mm')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Perfil guardado: {save_path}")
    else:
        plt.show()


def visualize_3d_isodose(dose_map, levels=[10, 30, 50, 70, 90, 95], save_path=None):
    """Visualiza curvas de isodosis 3D."""
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalizar
    dose_norm = (dose_map / np.max(dose_map)) * 100
    
    # Crear meshgrid
    z, y, x = np.indices(dose_map.shape)
    
    colors = cm.jet(np.linspace(0, 1, len(levels)))
    
    for level, color in zip(levels, colors):
        # Contornos 3D
        mask = (dose_norm >= level-2) & (dose_norm <= level+2)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], z[mask], 
                      c=[color], marker='.', s=1, alpha=0.3,
                      label=f'{level}%')
    
    ax.set_xlabel('X [voxels]')
    ax.set_ylabel('Y [voxels]')
    ax.set_zlabel('Z [voxels]')
    ax.set_title('Distribución 3D de Isodosis')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Isodosis 3D guardada: {save_path}")
    else:
        plt.show()


def main():
    """Script de visualización."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualización de dosis')
    parser.add_argument('--dose', type=str, required=True, help='Archivo de dosis')
    parser.add_argument('--plot-type', type=str, default='pdd', 
                       choices=['pdd', 'profile', 'isodose'])
    parser.add_argument('--output', type=str, help='Archivo de salida')
    
    args = parser.parse_args()
    
    # Cargar dosis
    dose_image = sitk.ReadImage(args.dose)
    dose_array = sitk.GetArrayFromImage(dose_image)
    
    if args.plot_type == 'pdd':
        plot_pdd_curve(dose_array, save_path=args.output)
    elif args.plot_type == 'profile':
        depth_idx = dose_array.shape[0] // 2
        plot_beam_profile(dose_array, depth_idx, save_path=args.output)
    elif args.plot_type == 'isodose':
        visualize_3d_isodose(dose_array, save_path=args.output)


if __name__ == '__main__':
    main()
