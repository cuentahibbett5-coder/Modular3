#!/usr/bin/env python3
"""
Convierte phase space ROOT a formato numpy (.npz)
"""

import uproot
import numpy as np
import argparse
from pathlib import Path


def convert_root_to_numpy(root_file, output_file=None):
    """
    Lee phase space ROOT y guarda en formato numpy comprimido.
    
    Args:
        root_file: Path al archivo .root
        output_file: Path de salida .npz (si None, usa mismo nombre)
    """
    root_path = Path(root_file)
    
    if output_file is None:
        output_file = root_path.with_suffix('.npz')
    
    print(f"📂 Leyendo: {root_file}")
    
    with uproot.open(root_file) as f:
        tree = f[f.keys()[0]]
        n_particles = tree.num_entries
        
        print(f"📊 Partículas totales: {n_particles:,}")
        print(f"🔄 Convirtiendo a numpy...")
        
        # Leer todos los datos
        data = tree.arrays(library='np')
        
        # Extraer arrays relevantes
        phsp_data = {
            'n_particles': n_particles,
            'energy': data['KineticEnergy'].astype(np.float32),  # MeV
            'weight': data['Weight'].astype(np.float32),
            'position_x': data['PrePosition_X'].astype(np.float32),  # mm
            'position_y': data['PrePosition_Y'].astype(np.float32),
            'position_z': data['PrePosition_Z'].astype(np.float32),
            'direction_x': data['PreDirection_X'].astype(np.float32),
            'direction_y': data['PreDirection_Y'].astype(np.float32),
            'direction_z': data['PreDirection_Z'].astype(np.float32),
            'pdg_code': data['PDGCode'].astype(np.int32),
        }
        
        # Estadísticas
        print(f"\n📋 Estadísticas:")
        print(f"   Energía [MeV]: {phsp_data['energy'].min():.3f} - {phsp_data['energy'].max():.3f}")
        print(f"   Media: {phsp_data['energy'].mean():.3f} MeV")
        
        unique_pdg, counts = np.unique(phsp_data['pdg_code'], return_counts=True)
        print(f"\n⚛️  Composición:")
        for pdg, count in zip(unique_pdg, counts):
            name = {-11: 'Positrón', 11: 'Electrón', 22: 'Fotón'}.get(pdg, f'PDG={pdg}')
            print(f"   {name}: {count:,} ({100*count/n_particles:.2f}%)")
        
        print(f"\n💾 Guardando: {output_file}")
        np.savez_compressed(output_file, **phsp_data)
        
        # Verificar tamaño
        size_mb = Path(output_file).stat().st_size / (1024**2)
        print(f"✅ Archivo generado: {size_mb:.1f} MB")
        
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Convierte phase space ROOT a numpy (.npz)'
    )
    parser.add_argument('--input', required=True, help='Archivo ROOT de entrada')
    parser.add_argument('--output', help='Archivo NPZ de salida (opcional)')
    
    args = parser.parse_args()
    
    convert_root_to_numpy(args.input, args.output)


if __name__ == '__main__':
    main()
