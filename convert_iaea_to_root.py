#!/usr/bin/env python3
"""
Convertidor IAEA PHSP → ROOT (Versión 2)

Lee el archivo phase-space IAEA correctamente según especificación:
- Formato: X Y Z U V W Energy(MeV) History ILB
- Byte order: Little-endian (code 1234)
- Record length: 37 bytes (7 floats + 2 ints + 1 byte padding)

Preserva TODAS las partículas:
- Fotones (γ)
- Electrones (e⁻)
- Positrones (e⁺)
"""

import argparse
import struct
import re
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import uproot
import awkward as ak


class IAEAHeader:
    """Parser del header IAEA."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self._parse()
    
    def _parse(self):
        """Parse el archivo header IAEA."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Extraer valores
        def extract_int(key):
            match = re.search(rf'\${key}:\s*\n\s*(\d+)', content)
            return int(match.group(1)) if match else None
        
        # Información crítica
        self.record_length = extract_int('RECORD_LENGTH')
        self.byte_order_code = extract_int('BYTE_ORDER')
        self.n_photons = extract_int('PHOTONS')
        self.n_electrons = extract_int('ELECTRONS')
        self.n_positrons = extract_int('POSITRONS')
        self.n_total = extract_int('PARTICLES')
        
        # Byte order
        if self.byte_order_code == 1234:
            self.byte_order = '<'  # Little-endian
        elif self.byte_order_code == 4321:
            self.byte_order = '>'  # Big-endian
        else:
            raise ValueError(f"Byte order desconocido: {self.byte_order_code}")
    
    def __repr__(self):
        return (
            f"IAEAHeader("
            f"record_len={self.record_length}, "
            f"photons={self.n_photons:,}, "
            f"electrons={self.n_electrons:,}, "
            f"positrons={self.n_positrons:,}, "
            f"total={self.n_total:,})"
        )


def decode_ilb(ilb: int) -> int:
    """
    Decodificar el campo ILB para obtener el tipo de partícula.
    
    En PENELOPE:
    - Bits 0-1: Generation number or particle id (LSB)
    - Bits 1-2: Particle type (1=e-, 2=e+, 3=γ)
    - Higher bits: Transport information
    """
    particle_code = (ilb >> 1) & 0x03
    
    # Mapear a PDG codes
    pid_map = {
        1: 11,      # electron (e-)
        2: -11,     # positron (e+)
        3: 22,      # photon (γ)
    }
    
    return pid_map.get(particle_code, 0)


def read_iaea_phsp(header: IAEAHeader, phsp_filepath: str, 
                   max_particles: int = None, verbose: bool = False) -> Tuple:
    """
    Lee el archivo PHSP IAEA binario.
    
    Returns:
        Tuple de arrays numpy: (x, y, z, dx, dy, dz, E, w, pid)
    """
    
    if verbose:
        print(f"📖 Leyendo PHSP IAEA: {phsp_filepath}")
        print(f"   Esperadas: {header.n_total:,} partículas")
        print(f"   Record length: {header.record_length} bytes")
        byte_order_str = "little-endian" if header.byte_order == '<' else "big-endian"
        print(f"   Byte order: {header.byte_order} ({byte_order_str})")
    
    n_to_read = min(max_particles, header.n_total) if max_particles else header.n_total
    
    # Formato: 7 floats (X Y Z U V W E) + 2 ints (History ILB) = 36 bytes + 1 padding
    fmt = header.byte_order + '7f2i'
    
    # Preallocar arrays
    x_arr = np.zeros(n_to_read, dtype=np.float32)
    y_arr = np.zeros(n_to_read, dtype=np.float32)
    z_arr = np.zeros(n_to_read, dtype=np.float32)
    dx_arr = np.zeros(n_to_read, dtype=np.float32)
    dy_arr = np.zeros(n_to_read, dtype=np.float32)
    dz_arr = np.zeros(n_to_read, dtype=np.float32)
    E_arr = np.zeros(n_to_read, dtype=np.float32)
    w_arr = np.ones(n_to_read, dtype=np.float32)  # Todos peso 1
    pid_arr = np.zeros(n_to_read, dtype=np.int32)
    
    # Contadores por tipo
    n_by_type = {22: 0, 11: 0, -11: 0, 0: 0}
    
    if verbose:
        print(f"\n🔄 Leyendo {n_to_read:,} partículas...\n")
        start_time = time.time()
    
    with open(phsp_filepath, 'rb') as f:
        for i in range(n_to_read):
            if verbose and i % 1000000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (n_to_read - i) / rate
                print(f"   [{i:,} / {n_to_read:,}]  "
                      f"{rate/1e6:.1f}M part/s  "
                      f"ETA {eta/60:.1f} min")
            
            # Leer registro (descartar último byte de padding)
            data = f.read(header.record_length)
            if len(data) < header.record_length:
                print(f"⚠️  Fin de archivo en registro {i}")
                # Truncar arrays
                x_arr = x_arr[:i]
                y_arr = y_arr[:i]
                z_arr = z_arr[:i]
                dx_arr = dx_arr[:i]
                dy_arr = dy_arr[:i]
                dz_arr = dz_arr[:i]
                E_arr = E_arr[:i]
                w_arr = w_arr[:i]
                pid_arr = pid_arr[:i]
                break
            
            # Decodificar registro (sin el byte de padding)
            try:
                values = struct.unpack(fmt, data[:36])
            except struct.error as e:
                print(f"❌ Error decodificando registro {i}: {e}")
                break
            
            x, y, z, u, v, w, energy, history, ilb = values
            
            # Normalizar dirección (en caso de necesidad)
            # Los valores u, v, w son SLOPES (tangentes), no cosenos
            # Para convertir a cosenos: dx = u/sqrt(1+u²+v²+w²), etc.
            # Pero OpenGate espera cosenos, así que normalizar a 1
            dir_norm = np.sqrt(u*u + v*v + w*w)
            if dir_norm > 0:
                u, v, w = u/dir_norm, v/dir_norm, w/dir_norm
            
            # Decodificar tipo de partícula
            pid = decode_ilb(ilb)
            
            # Almacenar (energía está en MeV)
            x_arr[i] = x
            y_arr[i] = y
            z_arr[i] = z
            dx_arr[i] = u
            dy_arr[i] = v
            dz_arr[i] = w
            E_arr[i] = energy
            w_arr[i] = 1.0
            pid_arr[i] = pid
            
            n_by_type[pid] += 1
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n✅ Leídas {i+1:,} partículas en {elapsed:.1f}s")
        print(f"\n   Composición:")
        print(f"      Fotones (γ): {n_by_type[22]:,}")
        print(f"      Electrones (e⁻): {n_by_type[11]:,}")
        print(f"      Positrones (e⁺): {n_by_type[-11]:,}")
        print(f"      Desconocidos: {n_by_type[0]:,}")
    
    return x_arr, y_arr, z_arr, dx_arr, dy_arr, dz_arr, E_arr, w_arr, pid_arr


def write_root_phsp(output_path: str, x, y, z, dx, dy, dz, E, w, pid, verbose: bool = False):
    """Escribe los datos en formato ROOT."""
    
    if verbose:
        print(f"💾 Escribiendo ROOT: {output_path}")
    
    # Crear diccionario con arrays
    data = {
        'x': ak.Array(x),
        'y': ak.Array(y),
        'z': ak.Array(z),
        'dx': ak.Array(dx),
        'dy': ak.Array(dy),
        'dz': ak.Array(dz),
        'E': ak.Array(E),
        'w': ak.Array(w),
        'pid': ak.Array(pid),
    }
    
    # Convertir a árbol ROOT
    with uproot.recreate(output_path) as f:
        f['phsp'] = data
    
    if verbose:
        print(f"✅ ROOT guardado")
        
        # Verificar
        f = uproot.open(output_path)
        tree = f['phsp']
        print(f"\n📊 Verificación:")
        print(f"   Árbol: phsp")
        print(f"   Entradas: {tree.num_entries:,}")
        
        print(f"\n   Estadísticas:")
        print(f"      X: [{x.min():.3f}, {x.max():.3f}] mm")
        print(f"      Y: [{y.min():.3f}, {y.max():.3f}] mm")
        print(f"      Z: [{z.min():.3f}, {z.max():.3f}] mm")
        print(f"      E: [{E.min():.3f}, {E.max():.3f}] MeV (mean={E.mean():.3f})")
        
        # Contar por tipo
        n_photons = np.sum(pid == 22)
        n_electrons = np.sum(pid == 11)
        n_positrons = np.sum(pid == -11)
        print(f"\n   Partículas:")
        print(f"      Fotones (γ): {n_photons:,}")
        print(f"      Electrones (e⁻): {n_electrons:,}")
        print(f"      Positrones (e⁺): {n_positrons:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir PHSP IAEA a ROOT (formato completo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test: 1 millón de partículas
  python convert_iaea_to_root.py \\
    --header data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAheader \\
    --phsp data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp \\
    --output data/IAEA/test_1M.root \\
    --max-particles 1000000

  # Conversión completa (29.4M partículas)
  python convert_iaea_to_root.py \\
    --header data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAheader \\
    --phsp data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp \\
    --output data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_full.root
        """
    )
    
    parser.add_argument('--header', required=True, help='Archivo header IAEA (.IAEAheader)')
    parser.add_argument('--phsp', required=True, help='Archivo PHSP IAEA binario (.IAEAphsp)')
    parser.add_argument('--output', required=True, help='Archivo ROOT de salida')
    parser.add_argument('--max-particles', type=int, default=None, 
                       help='Limitar a N partículas (para testing)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Salida detallada')
    
    args = parser.parse_args()
    
    # Validar archivos
    for path in [args.header, args.phsp]:
        if not Path(path).exists():
            print(f"❌ No existe: {path}")
            sys.exit(1)
    
    print("="*70)
    print("CONVERSOR IAEA → ROOT v2")
    print("="*70)
    print()
    
    # Parsear header
    header = IAEAHeader(args.header)
    print(f"Header IAEA: {header}\n")
    
    # Leer PHSP
    x, y, z, dx, dy, dz, E, w, pid = read_iaea_phsp(
        header, args.phsp, max_particles=args.max_particles, verbose=args.verbose
    )
    
    # Escribir ROOT
    write_root_phsp(args.output, x, y, z, dx, dy, dz, E, w, pid, verbose=args.verbose)
    
    print(f"\n✅ Conversión completada: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
