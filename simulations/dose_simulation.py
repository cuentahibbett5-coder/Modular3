#!/usr/bin/env python3
"""
Simulación de dosis con PhaseSpaceSource (IAEA PHSP).

Genera mapas de dosis 3D en un fantoma de agua usando partículas
de un archivo phase-space en formato ROOT.

Uso:
    python simulations/dose_simulation.py \
        --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
        --output output/test \
        --n-particles 100000 \
        --threads 4 \
        --seed 42

Dry-run (solo mostrar config):
    python simulations/dose_simulation.py --input ... --output ... --dry-run
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib
matplotlib.use('Agg')

import opengate as gate
import numpy as np


def run_simulation(args):
    """Ejecuta la simulación."""
    
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if not Path(args.input).exists():
        print(f"❌ ERROR: No se encuentra {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"SIMULACIÓN DOSIS - {args.job_id or 'LOCAL'}")
    print("=" * 60)
    print(f"  Input:      {args.input}")
    print(f"  Output:     {output_folder}")
    print(f"  Partículas: {args.n_particles:,}")
    print(f"  Threads:    {args.threads}")
    print(f"  Seed:       {args.seed}")
    print(f"  Spacing:    {args.spacing_xy}×{args.spacing_xy}×{args.spacing_z} mm")
    print("=" * 60)
    
    # Simulación
    sim = gate.Simulation()
    sim.random_seed = args.seed
    sim.number_of_threads = args.threads
    sim.output_dir = str(output_folder)
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'
    # Progreso siempre visible
    sim.progress_bar = True
    # Intervalo válido para el motor (inicio, fin)
    sim.run_timing_intervals = [(0, 1e9)]
    
    # World (aire) - suficiente para contener fuente y fantoma
    # El PHSP ya está en mm (Z≈785 mm)
    world = sim.world
    world.material = 'G4_AIR'
    
    # Geometría realista para radioterapia externa:
    # - PHSP está en Z ≈ 785 mm, partículas van hacia +Z (downstream)
    # - Gap de aire entre fuente y agua
    # - Agua downstream de la fuente
    
    gap_mm = args.gap  # Gap de aire en mm (default 50 mm = 5 cm)
    water_size = [300, 300, 300]  # mm (30×30×30 cm)
    
    # Superficie superior del agua = Z_fuente + gap
    # Z_fuente ≈ 785 mm (PHSP en mm)
    source_z = 785.0  # mm (posición del PHSP)
    water_top = source_z + gap_mm  # superficie superior del agua (downstream)
    water_center_z = water_top + water_size[2] / 2  # centro del agua

    # Ajustar tamaño de world en Z para cubrir fuente + agua + margen
    world.size = [
        max(500, water_size[0] + 200),
        max(500, water_size[1] + 200),
        max(2500, int(source_z + gap_mm + water_size[2] + 800))
    ]
    
    water = sim.add_volume('Box', 'water')
    water.size = water_size
    water.translation = [0, 0, water_center_z]
    water.material = 'G4_WATER'
    water.color = [0, 0.6, 0.8, 0.5]
    
    print(f"\n📦 Geometría:")
    print(f"   World: {world.size} mm (aire)")
    print(f"   Fuente PHSP: Z ≈ {source_z} mm")
    print(f"   Gap de aire: {gap_mm} mm ({gap_mm/10:.1f} cm)")
    print(f"   Agua: {water_size} mm")
    print(f"   Agua superficie: Z = {water_top:.1f} mm")
    print(f"   Agua centro: Z = {water_center_z:.1f} mm")
    
    # PhaseSpaceSource
    source = sim.add_source('PhaseSpaceSource', 'phsp')
    source.phsp_file = args.input
    source.particle = ''
    source.n = args.n_particles
    
    # Keys del ROOT (minúsculas)
    source.position_key_x = 'x'
    source.position_key_y = 'y'
    source.position_key_z = 'z'
    source.direction_key_x = 'dx'
    source.direction_key_y = 'dy'
    source.direction_key_z = 'dz'
    source.energy_key = 'E'
    source.weight_key = 'w'
    source.PDGCode_key = 'pid'
    
    if args.threads > 1:
        n_per_thread = args.n_particles // args.threads
        source.entry_start = [i * n_per_thread for i in range(args.threads)]
    
    print(f"📊 Source: {source.n:,} partículas")
    
    # DoseActor
    size_x = max(1, int(round(water_size[0] / args.spacing_xy)))
    size_y = max(1, int(round(water_size[1] / args.spacing_xy)))
    size_z = max(1, int(round(water_size[2] / args.spacing_z)))
    
    dose = sim.add_actor('DoseActor', 'dose')
    dose.attached_to = 'water'
    dose.size = [size_x, size_y, size_z]
    dose.spacing = [args.spacing_xy, args.spacing_xy, args.spacing_z]
    dose.output_filename = 'dose.mhd'
    dose.write_to_disk = True
    
    raw_mb = (size_x * size_y * size_z * 8) / (1024**2)
    print(f"📐 Dose grid: {dose.size} → {raw_mb:.1f} MB")
    
    # Stats
    sim.add_actor('SimulationStatisticsActor', 'stats')
    
    if args.dry_run:
        print(f"\n🔍 Dry-run completado.")
        return 0
    
    # Run
    print(f"\n🚀 Ejecutando...")
    t0 = time.time()
    try:
        sim.run()
        elapsed = time.time() - t0
        print(f"✅ Completado en {elapsed:.1f} s")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save metadata
    try:
        stats = sim.get_actor('stats')
        info = {
            'job_id': args.job_id,
            'input': args.input,
            'n_particles': args.n_particles,
            'threads': args.threads,
            'seed': args.seed,
            'grid_size': [size_x, size_y, size_z],
            'spacing_mm': [args.spacing_xy, args.spacing_xy, args.spacing_z],
            'runtime_seconds': float(elapsed),
        }
        # Try to get stats (API varies by version)
        try:
            info['events'] = int(stats.counts.event_count)
            info['tracks'] = int(stats.counts.track_count)
            info['steps'] = int(stats.counts.step_count)
        except Exception:
            try:
                info['events'] = int(stats.counts['event_count'])
                info['tracks'] = int(stats.counts.get('track_count', 0))
                info['steps'] = int(stats.counts.get('step_count', 0))
            except Exception:
                pass

        if 'events' in info:
            print(f"📊 Stats: events={info.get('events')}, tracks={info.get('tracks')}, steps={info.get('steps')}")
    except Exception as e:
        info = {'error': str(e)}
    
    with open(output_folder / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Guardado en {output_folder}")
    return 0


def main():
    p = argparse.ArgumentParser(description='Simulación dosis PHSP')
    p.add_argument('--input', '-i', required=True, help='PHSP ROOT file')
    p.add_argument('--output', '-o', required=True, help='Output directory')
    p.add_argument('--n-particles', '-n', type=int, default=1000000)
    p.add_argument('--threads', '-t', type=int, default=1)
    p.add_argument('--seed', '-s', type=int, default=42)
    p.add_argument('--spacing-xy', type=float, default=2.0, help='XY spacing mm')
    p.add_argument('--spacing-z', type=float, default=1.0, help='Z spacing mm')
    p.add_argument('--gap', type=float, default=50.0, help='Air gap between source and water (mm)')
    p.add_argument('--job-id', default=None)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    sys.exit(run_simulation(args))


if __name__ == '__main__':
    main()
