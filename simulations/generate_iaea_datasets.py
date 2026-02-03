#!/usr/bin/env python3
"""
Genera datasets de dosis con múltiples estadísticas para entrenamiento de IA.
Optimizado para el dataset IAEA de 29M partículas.

Este script genera pares de entrenamiento:
- Input: dosis con pocos eventos (ruidosa)
- Target: dosis con muchos eventos (limpia)
"""

import numpy as np
import opengate as gate
from pathlib import Path
import argparse
import time
import subprocess
import sys
import json


def load_iaea_phsp(npz_file):
    """Carga phase space en formato IAEA."""
    phsp = np.load(npz_file)
    
    return {
        'energy': phsp['energy'],  # MeV
        'pos_x': phsp['pos_x'] * 10,  # cm → mm
        'pos_y': phsp['pos_y'] * 10,
        'pos_z': phsp['pos_z'] * 10,
        'dir_u': phsp['dir_u'],
        'dir_v': phsp['dir_v'],
        'dir_w': phsp['dir_w'],
        'pdg': phsp['pdg'],
        'weight': phsp['weight'],
        'n_total': len(phsp['energy'])
    }


def run_dose_simulation(phsp_data, indices, output_dir, 
                        phantom_size_cm=30, dose_resolution_mm=2,
                        threads=1, seed=None):
    """
    Ejecuta simulación de dosis con un subconjunto de partículas.
    """
    n_particles = len(indices)
    
    # Extraer subset
    energy = phsp_data['energy'][indices]
    pos_x = phsp_data['pos_x'][indices]
    pos_y = phsp_data['pos_y'][indices]
    pos_z = phsp_data['pos_z'][indices]
    dir_u = phsp_data['dir_u'][indices]
    dir_v = phsp_data['dir_v'][indices]
    dir_w = phsp_data['dir_w'][indices]
    pdg = phsp_data['pdg'][indices]
    
    # Estadísticas
    unique_pdg, counts = np.unique(pdg, return_counts=True)
    particle_map = {-11: 'e+', 11: 'e-', 22: 'gamma'}
    
    print(f"\n{'='*60}")
    print(f"SIMULACIÓN: {n_particles:,} partículas")
    print(f"{'='*60}")
    for p, c in zip(unique_pdg, counts):
        name = particle_map.get(p, f'PDG={p}')
        print(f"   {name}: {c:,} ({100*c/n_particles:.1f}%)")
    
    # ================================================================
    # SIMULACIÓN
    # ================================================================
    sim = gate.Simulation()
    
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    
    sim.world.size = [200 * cm, 200 * cm, 200 * cm]
    sim.world.material = 'G4_AIR'
    
    # Phantom
    phantom = sim.add_volume('Box', 'water_phantom')
    phantom.size = [phantom_size_cm * cm] * 3
    phantom.material = 'G4_WATER'
    
    # Posición del phantom (5cm debajo del plano de fase)
    phsp_z_cm = pos_z[0] / 10.0  # mm → cm
    phantom_z_cm = phsp_z_cm - 5.0 - phantom_size_cm / 2
    phantom.translation = [0, 0, phantom_z_cm * cm]
    
    # Radio efectivo del haz
    r_mm = np.sqrt(pos_x**2 + pos_y**2)
    beam_radius_mm = float(np.percentile(r_mm, 99))
    
    print(f"   Phase space Z: {phsp_z_cm:.1f} cm")
    print(f"   Phantom centro Z: {phantom_z_cm:.1f} cm")
    print(f"   Radio del haz: {beam_radius_mm:.1f} mm")
    
    # Física
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'
    sim.physics_manager.set_production_cut('world', 'all', 1 * mm)
    sim.physics_manager.set_production_cut('water_phantom', 'all', 0.1 * mm)
    
    # ================================================================
    # FUENTES (una por tipo de partícula)
    # ================================================================
    for pdg_code in unique_pdg:
        if pdg_code not in particle_map:
            continue
        
        mask = pdg == pdg_code
        n_this = int(mask.sum())
        if n_this == 0:
            continue
        
        particle_name = particle_map[pdg_code]
        source = sim.add_source('GenericSource', f'source_{particle_name}')
        source.particle = particle_name
        source.n = n_this
        
        # Energía: histograma de 100 bins
        e_this = energy[mask]
        hist, bins = np.histogram(e_this, bins=100)
        source.energy.type = 'histogram'
        source.energy.histogram_weight = hist.tolist()
        source.energy.histogram_energy = ((bins[:-1] + bins[1:]) / 2).tolist()
        
        # Posición: disco
        source.position.type = 'disc'
        source.position.radius = beam_radius_mm * mm
        source.position.translation = [
            float(np.mean(pos_x[mask])) * mm,
            float(np.mean(pos_y[mask])) * mm,
            float(np.mean(pos_z[mask])) * mm
        ]
        
        # Dirección: focalizada al centro del phantom
        source.direction.type = 'focused'
        source.direction.focus_point = [0, 0, phantom_z_cm * cm]
    
    # ================================================================
    # DOSE ACTOR
    # ================================================================
    dose_actor = sim.add_actor('DoseActor', 'dose')
    dose_actor.attached_to = 'water_phantom'
    n_voxels = int(phantom_size_cm * 10 / dose_resolution_mm)
    dose_actor.size = [n_voxels] * 3
    dose_actor.spacing = [dose_resolution_mm * mm] * 3
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dose_actor.output_filename = str(output_path / 'dose.mhd')
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    
    # Stats
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    
    # Config
    sim.random_seed = seed if seed else 'auto'
    sim.number_of_threads = threads
    
    # Run
    print(f"\n▶️  Ejecutando ({threads} threads)...")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    
    print(f"✅ {elapsed:.1f}s - Eventos: {stats.counts['events']:,}")
    
    return {
        'n_particles': n_particles,
        'events': stats.counts['events'],
        'time_s': elapsed
    }


def run_single_dataset(args):
    """Ejecuta una única simulación (para llamadas por subprocess)."""
    
    print(f"\n📂 Cargando: {args.phsp}")
    phsp = load_iaea_phsp(args.phsp)
    
    n_total = phsp['n_total']
    n_use = args.particles if args.particles != -1 else n_total
    
    # Generar índices (reproducibles si se da seed)
    if args.seed:
        np.random.seed(args.seed)
    
    if n_use < n_total:
        indices = np.random.choice(n_total, size=n_use, replace=False)
    else:
        indices = np.arange(n_total)
    
    result = run_dose_simulation(
        phsp, indices, args.output,
        phantom_size_cm=args.phantom_size,
        dose_resolution_mm=args.resolution,
        threads=args.threads,
        seed=args.seed
    )
    
    # Guardar metadata
    with open(Path(args.output) / 'metadata.json', 'w') as f:
        json.dump(result, f, indent=2)


def generate_training_pairs(phsp_file, output_dir, 
                            n_pairs=100,
                            noisy_particles=100000,
                            clean_particles=5000000,
                            phantom_size=30,
                            resolution=2,
                            threads=1):
    """
    Genera pares de entrenamiento (ruidoso, limpio) para la red de denoising.
    
    Cada par usa el MISMO subconjunto de partículas pero con diferente
    cantidad de submuestreo para el ruidoso.
    """
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERACIÓN DE PARES DE ENTRENAMIENTO")
    print(f"{'='*70}")
    print(f"Pares a generar: {n_pairs}")
    print(f"Partículas ruidoso: {noisy_particles:,}")
    print(f"Partículas limpio: {clean_particles:,}")
    print(f"Output: {output_base}")
    
    # Generar cada par usando subprocess (limitación de OpenGate)
    for i in range(n_pairs):
        print(f"\n{'='*70}")
        print(f"PAR {i+1}/{n_pairs}")
        print(f"{'='*70}")
        
        seed_base = 42 + i * 1000
        
        # Dataset LIMPIO (ground truth)
        clean_dir = output_base / f'pair_{i:04d}' / 'clean'
        cmd_clean = [
            sys.executable, __file__,
            '--single',
            '--phsp', str(phsp_file),
            '--output', str(clean_dir),
            '--particles', str(clean_particles),
            '--phantom-size', str(phantom_size),
            '--resolution', str(resolution),
            '--threads', str(threads),
            '--seed', str(seed_base)
        ]
        
        print(f"\n🔹 Generando LIMPIO ({clean_particles:,} partículas)...")
        subprocess.run(cmd_clean, check=True)
        
        # Dataset RUIDOSO (input para la red)
        noisy_dir = output_base / f'pair_{i:04d}' / 'noisy'
        cmd_noisy = [
            sys.executable, __file__,
            '--single',
            '--phsp', str(phsp_file),
            '--output', str(noisy_dir),
            '--particles', str(noisy_particles),
            '--phantom-size', str(phantom_size),
            '--resolution', str(resolution),
            '--threads', str(threads),
            '--seed', str(seed_base + 1)  # Diferente seed para variar el submuestreo
        ]
        
        print(f"\n🔹 Generando RUIDOSO ({noisy_particles:,} partículas)...")
        subprocess.run(cmd_noisy, check=True)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETADO: {n_pairs} pares generados")
    print(f"{'='*70}")


def generate_multi_stat_datasets(phsp_file, output_dir,
                                 stats_list=[50000, 100000, 500000, 1000000, 5000000, 'all'],
                                 phantom_size=30, resolution=2, threads=1):
    """
    Genera datasets con múltiples niveles de estadística.
    Similar al script anterior pero optimizado para 29M partículas.
    """
    output_base = Path(output_dir)
    
    # Cargar para obtener n_total
    phsp = load_iaea_phsp(phsp_file)
    n_total = phsp['n_total']
    del phsp  # Liberar memoria
    
    for stat in stats_list:
        if stat == 'all':
            n_particles = n_total
            label = f'full_{n_total}'
        else:
            n_particles = int(stat)
            if n_particles >= 1000000:
                label = f'{n_particles // 1000000}M'
            else:
                label = f'{n_particles // 1000}k'
        
        out_dir = output_base / f'dose_{label}'
        
        cmd = [
            sys.executable, __file__,
            '--single',
            '--phsp', str(phsp_file),
            '--output', str(out_dir),
            '--particles', str(n_particles) if stat != 'all' else '-1',
            '--phantom-size', str(phantom_size),
            '--resolution', str(resolution),
            '--threads', str(threads),
            '--seed', '42'
        ]
        
        print(f"\n{'='*70}")
        print(f"Generando: {label} ({n_particles:,} partículas)")
        print(f"{'='*70}")
        
        subprocess.run(cmd, check=True)
    
    print(f"\n✅ Todos los datasets generados en: {output_base}")


def main():
    parser = argparse.ArgumentParser(
        description='Genera datasets de dosis para entrenamiento de IA'
    )
    
    # Modo único (llamado por subprocess)
    parser.add_argument('--single', action='store_true',
                        help='Ejecutar una sola simulación')
    
    # Parámetros comunes
    parser.add_argument('--phsp', required=True, help='Phase space .npz (IAEA)')
    parser.add_argument('--output', required=True, help='Directorio de salida')
    parser.add_argument('--phantom-size', type=float, default=30, help='Tamaño phantom [cm]')
    parser.add_argument('--resolution', type=float, default=2, help='Resolución [mm]')
    parser.add_argument('--threads', type=int, default=1, help='Threads')
    
    # Para modo single
    parser.add_argument('--particles', type=int, default=-1, help='Partículas (-1=all)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Modos de generación batch
    parser.add_argument('--pairs', type=int, help='Generar N pares de entrenamiento')
    parser.add_argument('--noisy-particles', type=int, default=100000,
                        help='Partículas para dataset ruidoso')
    parser.add_argument('--clean-particles', type=int, default=5000000,
                        help='Partículas para dataset limpio')
    
    parser.add_argument('--multi-stat', action='store_true',
                        help='Generar datasets con múltiples estadísticas')
    parser.add_argument('--stats', nargs='+', default=['50k', '100k', '500k', '1M', '5M', 'all'],
                        help='Lista de estadísticas (ej: 50k 100k 1M all)')
    
    args = parser.parse_args()
    
    if args.single:
        # Modo single: ejecutar una simulación
        run_single_dataset(args)
    
    elif args.pairs:
        # Generar pares de entrenamiento
        generate_training_pairs(
            args.phsp, args.output,
            n_pairs=args.pairs,
            noisy_particles=args.noisy_particles,
            clean_particles=args.clean_particles,
            phantom_size=args.phantom_size,
            resolution=args.resolution,
            threads=args.threads
        )
    
    elif args.multi_stat:
        # Generar datasets con múltiples estadísticas
        stats_list = []
        for s in args.stats:
            if s == 'all':
                stats_list.append('all')
            elif s.endswith('M'):
                stats_list.append(int(s[:-1]) * 1000000)
            elif s.endswith('k'):
                stats_list.append(int(s[:-1]) * 1000)
            else:
                stats_list.append(int(s))
        
        generate_multi_stat_datasets(
            args.phsp, args.output,
            stats_list=stats_list,
            phantom_size=args.phantom_size,
            resolution=args.resolution,
            threads=args.threads
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
