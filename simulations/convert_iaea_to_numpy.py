#!/usr/bin/env python3
"""
Convierte archivo IAEA phase space (.IAEAphsp) a formato NumPy (.npz).

Formato IAEA (según header):
- RECORD_LENGTH: 37 bytes
- X, Y, Z, U, V, W, Weight: float32 (7 * 4 = 28 bytes)
- Extra longs: 2 * int32 (8 bytes)
- Signo de partícula: int8 (1 byte)
Total: 37 bytes

El tipo de partícula se codifica en ILB PENELOPE (extra long [1]):
- ILB % 10 = 1: electrón
- ILB % 10 = 2: fotón
- ILB % 10 = 3: positrón

Coordenadas IAEA:
- X, Y, Z en cm
- U, V, W son cosenos directores (dirección normalizada)
- Z = 78.45-78.5 cm según header (plano de captura)
"""

import numpy as np
from pathlib import Path
import argparse
import struct


def parse_iaea_header(header_file):
    """Lee el archivo .IAEAheader y extrae información relevante."""
    info = {}
    with open(header_file, 'r') as f:
        content = f.read()
    
    # Buscar campos clave
    lines = content.split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('$'):
            current_key = line[1:].rstrip(':')
        elif current_key and line and not line.startswith('//'):
            if current_key == 'RECORD_LENGTH':
                info['record_length'] = int(line)
            elif current_key == 'PARTICLES':
                info['n_particles'] = int(line)
            elif current_key == 'PHOTONS':
                info['n_photons'] = int(line)
            elif current_key == 'ELECTRONS':
                info['n_electrons'] = int(line)
            elif current_key == 'POSITRONS':
                info['n_positrons'] = int(line)
            elif current_key == 'BYTE_ORDER':
                info['byte_order'] = line
    
    return info


def read_iaea_phsp(phsp_file, header_info, max_particles=None):
    """
    Lee archivo binario IAEA phase space.
    
    Formato del registro (37 bytes) para este archivo PENELOPE:
    - 1 byte: tipo de partícula (1=e-, 2=gamma, 3=e+)
    - 7 floats (28 bytes):
        - X: posición X en cm
        - Y: posición Y en cm  
        - E_signed: energía en MeV, signo indica dirección de W
        - Z: posición Z en cm (constante ~78.45)
        - U: coseno director en X
        - V: coseno director en Y
        - Weight: peso estadístico
    - 2 int32 (8 bytes): extra longs (history number, ILB)
    
    W se calcula como: sign(E) * sqrt(1 - U² - V²)
    """
    record_length = header_info.get('record_length', 37)
    n_particles = header_info.get('n_particles', 0)
    
    if max_particles:
        n_particles = min(n_particles, max_particles)
    
    print(f"Leyendo {n_particles:,} partículas...")
    print(f"Record length: {record_length} bytes")
    
    # Arrays para almacenar datos
    pos_x = np.zeros(n_particles, dtype=np.float32)
    pos_y = np.zeros(n_particles, dtype=np.float32)
    pos_z = np.zeros(n_particles, dtype=np.float32)
    dir_u = np.zeros(n_particles, dtype=np.float32)
    dir_v = np.zeros(n_particles, dtype=np.float32)
    dir_w = np.zeros(n_particles, dtype=np.float32)
    energy = np.zeros(n_particles, dtype=np.float32)
    weight = np.zeros(n_particles, dtype=np.float32)
    pdg = np.zeros(n_particles, dtype=np.int32)
    
    # Formato: '<' = little endian, 'b' = int8, 'f' = float32, 'i' = int32
    record_format = '<b7f2i'  # 1 + 28 + 8 = 37 bytes
    
    with open(phsp_file, 'rb') as f:
        for i in range(n_particles):
            if i % 5000000 == 0 and i > 0:
                print(f"  Procesadas {i:,} partículas...")
            
            data = f.read(record_length)
            if len(data) < record_length:
                print(f"  Fin de archivo en partícula {i}")
                break
            
            try:
                # Desempaquetar registro
                values = struct.unpack(record_format, data)
                ptype = values[0]
                x = values[1]
                y = values[2]
                e_signed = values[3]
                z = values[4]
                u = values[5]
                v = values[6]
                wt = values[7]
                
                # Calcular W desde U, V y signo de energía
                e = abs(e_signed)
                w_sign = 1 if e_signed > 0 else -1
                w = w_sign * np.sqrt(max(0, 1 - u**2 - v**2))
                
                pos_x[i] = x
                pos_y[i] = y
                pos_z[i] = z
                dir_u[i] = u
                dir_v[i] = v
                dir_w[i] = w
                energy[i] = e
                weight[i] = wt
                
                # Tipo de partícula desde byte inicial
                # Para este archivo PENELOPE específico:
                # 1=fotón (gamma), 2=electrón, 3=positrón
                # (verificado contra estadísticas del header)
                if ptype == 1:  # Fotón
                    pdg[i] = 22
                elif ptype == 2:  # Electrón
                    pdg[i] = 11
                elif ptype == 3:  # Positrón
                    pdg[i] = -11
                else:
                    pdg[i] = 0  # Desconocido
                    
            except struct.error as e:
                print(f"  Error en partícula {i}: {e}")
                break
    
    # Recortar arrays si terminamos antes
    actual_count = i + 1 if i < n_particles else n_particles
    
    return {
        'pos_x': pos_x[:actual_count],
        'pos_y': pos_y[:actual_count],
        'pos_z': pos_z[:actual_count],
        'dir_u': dir_u[:actual_count],
        'dir_v': dir_v[:actual_count],
        'dir_w': dir_w[:actual_count],
        'energy': energy[:actual_count],
        'weight': weight[:actual_count],
        'pdg': pdg[:actual_count]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convierte archivo IAEA phase space a NumPy npz'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Archivo .IAEAphsp de entrada')
    parser.add_argument('--output', '-o', default=None,
                        help='Archivo .npz de salida (default: mismo nombre)')
    parser.add_argument('--max-particles', '-n', type=int, default=None,
                        help='Número máximo de partículas a leer')
    
    args = parser.parse_args()
    
    phsp_file = Path(args.input)
    header_file = phsp_file.with_suffix('.IAEAheader')
    
    if not phsp_file.exists():
        print(f"Error: No existe {phsp_file}")
        return
    
    if not header_file.exists():
        print(f"Error: No existe header {header_file}")
        return
    
    # Leer header
    print(f"\n📋 Leyendo header: {header_file.name}")
    header_info = parse_iaea_header(header_file)
    print(f"   Partículas totales: {header_info.get('n_particles', 'N/A'):,}")
    print(f"   Fotones: {header_info.get('n_photons', 'N/A'):,}")
    print(f"   Electrones: {header_info.get('n_electrons', 'N/A'):,}")
    print(f"   Positrones: {header_info.get('n_positrons', 'N/A'):,}")
    
    # Leer phase space
    print(f"\n📖 Leyendo phase space: {phsp_file.name}")
    data = read_iaea_phsp(phsp_file, header_info, args.max_particles)
    
    # Estadísticas
    n_total = len(data['pdg'])
    n_photons = np.sum(data['pdg'] == 22)
    n_electrons = np.sum(data['pdg'] == 11)
    n_positrons = np.sum(data['pdg'] == -11)
    
    print(f"\n📊 Estadísticas leídas:")
    print(f"   Total: {n_total:,}")
    print(f"   Fotones (γ): {n_photons:,} ({100*n_photons/n_total:.1f}%)")
    print(f"   Electrones (e-): {n_electrons:,} ({100*n_electrons/n_total:.1f}%)")
    print(f"   Positrones (e+): {n_positrons:,} ({100*n_positrons/n_total:.1f}%)")
    
    # Información de posición
    print(f"\n📍 Rango de posiciones (cm):")
    print(f"   X: [{data['pos_x'].min():.2f}, {data['pos_x'].max():.2f}]")
    print(f"   Y: [{data['pos_y'].min():.2f}, {data['pos_y'].max():.2f}]")
    print(f"   Z: [{data['pos_z'].min():.2f}, {data['pos_z'].max():.2f}]")
    
    # Guardar npz
    output_file = Path(args.output) if args.output else phsp_file.with_suffix('.npz')
    
    print(f"\n💾 Guardando: {output_file}")
    np.savez_compressed(
        output_file,
        pos_x=data['pos_x'],
        pos_y=data['pos_y'],
        pos_z=data['pos_z'],
        dir_u=data['dir_u'],
        dir_v=data['dir_v'],
        dir_w=data['dir_w'],
        energy=data['energy'],
        weight=data['weight'],
        pdg=data['pdg'],
        # Metadata
        source='IAEA',
        original_file=str(phsp_file.name),
        coordinate_unit='cm',
        direction_unit='cosines',
        energy_unit='MeV'
    )
    
    output_size = output_file.stat().st_size / (1024 * 1024)
    print(f"   Tamaño: {output_size:.1f} MB")
    print(f"\n✅ Conversión completada!")


if __name__ == '__main__':
    main()
