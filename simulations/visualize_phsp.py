"""
Script simple para visualizar contenido del phase space ROOT
"""

import uproot
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import numpy as np

# Abrir archivo
file_path = "data/phase_space/varian_iaea_full.root"
print(f"Abriendo: {file_path}")

with uproot.open(file_path) as f:
    tree = f["PhaseSpace"]
    
    print(f"\n📊 Total de partículas: {tree.num_entries:,}")
    print(f"\n📋 Ramas disponibles:")
    for branch in tree.keys():
        print(f"   - {branch}")
    
    # Leer primeras 100k entradas para plotting rápido
    n_plot = min(100000, tree.num_entries)
    print(f"\n🎨 Graficando primeras {n_plot:,} partículas...")
    
    data = tree.arrays(library="np", entry_stop=n_plot)
    
    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Varian Clinac 6MV Phase Space - {n_plot:,} partículas', fontsize=16)
    
    # 1. Distribución de energía
    ax = axes[0, 0]
    ax.hist(data['Ekine'], bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Energía cinética (MeV)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Espectro de energía')
    ax.grid(True, alpha=0.3)
    
    # 2. Tipos de partículas
    ax = axes[0, 1]
    pdg_codes = data['PDGCode']
    unique, counts = np.unique(pdg_codes, return_counts=True)
    labels = []
    for code in unique:
        if code == 22:
            labels.append('Fotones')
        elif code == 11:
            labels.append('Electrones')
        elif code == -11:
            labels.append('Positrones')
        else:
            labels.append(f'PDG {code}')
    ax.bar(labels, counts)
    ax.set_ylabel('Número de partículas')
    ax.set_title('Composición')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Posición XY
    ax = axes[0, 2]
    ax.hist2d(data['PrePosition_X'], data['PrePosition_Y'], bins=50, cmap='viridis')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Distribución espacial XY')
    ax.set_aspect('equal')
    
    # 4. Energía por tipo de partícula
    ax = axes[1, 0]
    for code, label in zip([22, 11, -11], ['Fotones', 'Electrones', 'Positrones']):
        mask = pdg_codes == code
        if np.any(mask):
            ax.hist(data['Ekine'][mask], bins=50, alpha=0.5, label=label)
    ax.set_xlabel('Energía (MeV)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Espectro por tipo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Dirección (ángulos)
    ax = axes[1, 1]
    dx = data['PreDirectionLocal_X']
    dy = data['PreDirectionLocal_Y']
    dz = data['PreDirectionLocal_Z']
    theta = np.arccos(dz) * 180 / np.pi  # ángulo respecto a Z
    ax.hist(theta, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Ángulo θ respecto a Z (grados)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución angular')
    ax.grid(True, alpha=0.3)
    
    # 6. Estadísticas
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
Estadísticas (primeras {n_plot:,})

Energía:
  Min:    {data['Ekine'].min():.3f} MeV
  Max:    {data['Ekine'].max():.3f} MeV
  Media:  {data['Ekine'].mean():.3f} MeV
  
Posición X:
  Rango:  [{data['PrePosition_X'].min():.2f}, {data['PrePosition_X'].max():.2f}] cm
  
Posición Y:
  Rango:  [{data['PrePosition_Y'].min():.2f}, {data['PrePosition_Y'].max():.2f}] cm
  
Posición Z:
  Rango:  [{data['PrePosition_Z'].min():.2f}, {data['PrePosition_Z'].max():.2f}] cm
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Guardar
    output_file = "results/varian_phsp_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Gráfica guardada en: {output_file}")
    print(f"   Abre el archivo PNG para ver los datos")

