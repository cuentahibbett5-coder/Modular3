#!/usr/bin/env python3
"""
Calcula dosis leyendo phase space desde archivo numpy (.npz)
Usa GenericSource para recrear las partículas con sus propiedades exactas.
"""

import opengate as gate
from pathlib import Path
import numpy as np
import argparse


def dose_from_numpy_phsp(npz_file, output_dir, phantom_size_cm=30,
                         dose_resolution_mm=2, n_particles=None, threads=4):
    """
    Calcula dosis leyendo phase space desde numpy y usando GenericSource.
    """
    
    # Cargar phase space
    print(f"📂 Cargando phase space: {npz_file}")
    phsp = np.load(npz_file)
    
    n_total = int(phsp['n_particles'])
    n_use = min(n_total, n_particles) if n_particles else n_total
    
    print(f"📊 Partículas: {n_use:,} de {n_total:,}")
    
    # Extraer datos (limitar a n_use)
    energy = phsp['energy'][:n_use]
    pos_x = phsp['position_x'][:n_use]
    pos_y = phsp['position_y'][:n_use]
    pos_z = phsp['position_z'][:n_use]
    dir_x = phsp['direction_x'][:n_use]
    dir_y = phsp['direction_y'][:n_use]
    dir_z = phsp['direction_z'][:n_use]
    pdg = phsp['pdg_code'][:n_use]
    
    # Estadísticas
    unique_pdg, counts = np.unique(pdg, return_counts=True)
    print(f"\n📋 Composición:")
    for p, c in zip(unique_pdg, counts):
        name = {-11: 'e+', 11: 'e-', 22: 'gamma'}.get(p, f'PDG={p}')
        print(f"   {name}: {c:,} ({100*c/n_use:.1f}%)")
    
    print(f"\n⚡ Energía: {energy.min():.3f} - {energy.max():.3f} MeV")
    print(f"📐 Dirección Z media: {dir_z.mean():.3f}")
    
    # ================================================================
    # SIMULACIÓN
    # ================================================================
    sim = gate.Simulation()
    
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    
    # Mundo
    sim.world.size = [200 * cm, 200 * cm, 200 * cm]
    sim.world.material = 'G4_AIR'
    
    # Phantom
    phantom = sim.add_volume('Box', 'water_phantom')
    phantom.size = [phantom_size_cm * cm] * 3
    phantom.material = 'G4_WATER'
    phantom.color = [0, 0, 1, 0.3]
    
    # Posición: 5cm arriba del phase space (partículas van en -Z)
    phsp_z_cm = pos_z[0] / 10.0  # mm → cm
    phantom_z_cm = phsp_z_cm - 5.0
    phantom.translation = [0, 0, phantom_z_cm * cm]
    
    print(f"\n🎯 Geometría:")
    print(f"   Phase space Z: {phsp_z_cm:.2f} cm")
    print(f"   Phantom Z: {phantom_z_cm:.2f} cm")
    
    # Física
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'
    sim.physics_manager.set_production_cut('world', 'all', 1 * mm)
    sim.physics_manager.set_production_cut('water_phantom', 'all', 0.1 * mm)
    
    # ================================================================
    # FUENTES: separar por tipo de partícula
    # ================================================================
    particle_map = {-11: 'e+', 11: 'e-', 22: 'gamma'}
    
    for pdg_code in unique_pdg:
        if pdg_code not in particle_map:
            continue
        
        mask = pdg == pdg_code
        n_this = mask.sum()
        
        if n_this == 0:
            continue
        
        particle_name = particle_map[pdg_code]
        source = sim.add_source('GenericSource', f'source_{particle_name}')
        source.particle = particle_name
        source.n = int(n_this)
        
        # Energía: usar histograma discreto
        energies_this = energy[mask]
        source.energy.type = 'histogram'
        # Crear bins de energía
        hist, bins = np.histogram(energies_this, bins=100)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        source.energy.histogram_weight = hist.tolist()
        source.energy.histogram_energy = bin_centers.tolist()
        
        # Posición: usar el promedio (todas en mismo plano)
        source.position.type = 'sphere'
        source.position.radius = 0.001 * mm  # Punto
        source.position.translation = [
            float(np.mean(pos_x[mask])),
            float(np.mean(pos_y[mask])),
            float(np.mean(pos_z[mask]))
        ]
        
        # Dirección: usar distribución promedio
        # GenericSource no permite lista de direcciones, usamos momentum promedio
        source.direction.type = 'momentum'
        source.direction.momentum = [
            float(np.mean(dir_x[mask])),
            float(np.mean(dir_y[mask])),
            float(np.mean(dir_z[mask]))
        ]
        
        print(f"   Fuente {particle_name}: {n_this:,} partículas")
    
    # ================================================================
    # DOSE ACTOR
    # ================================================================
    dose_actor = sim.add_actor('DoseActor', 'dose')
    dose_actor.attached_to = 'water_phantom'
    dose_actor.size = [int(phantom_size_cm * 10 / dose_resolution_mm)] * 3
    dose_actor.spacing = [dose_resolution_mm * mm] * 3
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dose_actor.output_filename = str(output_path / 'dose.mhd')
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    
    # Stats
    stats = sim.add_actor('SimulationStatisticsActor', 'stats')
    stats.track_types_flag = True
    
    # Config
    sim.random_seed = 'auto'
    sim.number_of_threads = threads
    
    # Run
    print(f"\n▶️  Ejecutando ({threads} threads)...")
    sim.run()
    
    print(f"\n✅ COMPLETADO")
    print(f"📊 Eventos: {stats.counts['events']:,}")
    print(f"💾 Output: {output_dir}")
    
    return sim


def main():
    parser = argparse.ArgumentParser(
        description='Calcular dosis desde phase space numpy'
    )
    parser.add_argument('--phsp', required=True, help='Phase space .npz')
    parser.add_argument('--output', required=True, help='Directorio de salida')
    parser.add_argument('--phantom-size', type=float, default=30, help='Tamaño phantom [cm]')
    parser.add_argument('--resolution', type=float, default=2, help='Resolución dosis [mm]')
    parser.add_argument('--particles', type=int, help='Máximo de partículas a usar')
    parser.add_argument('--threads', type=int, default=4, help='Número de threads')
    
    args = parser.parse_args()
    
    dose_from_numpy_phsp(
        args.phsp,
        args.output,
        phantom_size_cm=args.phantom_size,
        dose_resolution_mm=args.resolution,
        n_particles=args.particles,
        threads=args.threads
    )


if __name__ == '__main__':
    main()
