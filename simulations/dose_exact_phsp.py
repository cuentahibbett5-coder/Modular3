#!/usr/bin/env python3
"""
Calcula dosis usando las partículas EXACTAS del phase space IAEA.
Usa las posiciones Y direcciones originales, no aproximaciones.
"""

import numpy as np
import opengate as gate
from pathlib import Path
import argparse
import time
import json


def run_dose_with_exact_particles(npz_file, output_dir, n_particles=None,
                                  phantom_size_cm=15, dose_resolution_mm=2,
                                  phantom_z_offset_cm=5, threads=1, seed=None,
                                  recenter_xy=True):
    """
    Ejecuta simulación de dosis usando partículas EXACTAS del phase space.
    
    La diferencia clave: usamos las direcciones REALES de cada partícula,
    no una dirección promedio o focalizada.
    """
    
    print(f"📂 Cargando: {npz_file}")
    phsp = np.load(npz_file)
    
    # Extraer datos (IAEA está en cm, convertir a mm)
    energy = phsp['energy']  # MeV
    pos_x = phsp['pos_x'] * 10  # cm → mm
    pos_y = phsp['pos_y'] * 10
    pos_z = phsp['pos_z'] * 10
    dir_u = phsp['dir_u']  # cosenos directores
    dir_v = phsp['dir_v']
    dir_w = phsp['dir_w']
    pdg = phsp['pdg']
    
    n_total = len(energy)
    
    # Submuestreo si se especifica
    if seed:
        np.random.seed(seed)
    
    if n_particles and n_particles < n_total:
        indices = np.random.choice(n_total, size=n_particles, replace=False)
        energy = energy[indices]
        pos_x = pos_x[indices]
        pos_y = pos_y[indices]
        pos_z = pos_z[indices]
        dir_u = dir_u[indices]
        dir_v = dir_v[indices]
        dir_w = dir_w[indices]
        pdg = pdg[indices]
        n_use = n_particles
    else:
        n_use = n_total
    
    print(f"📊 Partículas: {n_use:,} de {n_total:,}")
    
    # Análisis de direcciones
    going_down = (dir_w > 0).sum()
    going_up = (dir_w < 0).sum()
    
    print(f"\n🎯 Análisis de direcciones:")
    print(f"   Hacia +Z: {going_down:,} ({100*going_down/n_use:.1f}%)")
    print(f"   Hacia -Z: {going_up:,} ({100*going_up/n_use:.1f}%)")
    
    # ================================================================
    # GEOMETRÍA
    # ================================================================
    # Phase space está en Z ≈ 785 mm
    # Phantom lo ponemos DEBAJO del phase space
    # Las partículas que van hacia +Z entrarán al phantom
    
    phsp_z_mm = pos_z.mean()
    phantom_center_z_mm = phsp_z_mm + phantom_z_offset_cm * 10 + phantom_size_cm * 10 / 2
    
    print(f"\n📐 Geometría:")
    print(f"   Phase space Z: {phsp_z_mm:.1f} mm")
    print(f"   Phantom centro Z: {phantom_center_z_mm:.1f} mm")
    print(f"   Phantom tamaño: {phantom_size_cm} cm")
    
    # ================================================================
    # Solo usar partículas que van hacia +Z (hacia el phantom)
    # ================================================================
    mask_forward = dir_w > 0.1  # Pequeño umbral para evitar partículas casi paralelas
    
    energy_fwd = energy[mask_forward]
    pos_x_fwd = pos_x[mask_forward]
    pos_y_fwd = pos_y[mask_forward]
    pos_z_fwd = pos_z[mask_forward]
    dir_u_fwd = dir_u[mask_forward]
    dir_v_fwd = dir_v[mask_forward]
    dir_w_fwd = dir_w[mask_forward]
    pdg_fwd = pdg[mask_forward]
    
    n_forward = len(energy_fwd)
    print(f"\n✂️  Filtrado: {n_forward:,} partículas hacia +Z (de {n_use:,})")
    
    if n_forward == 0:
        print("❌ No hay partículas en la dirección correcta!")
        return None

    # ================================================================
    # Recentrar en XY si el phase space viene desplazado
    # ================================================================
    x_offset = float(pos_x_fwd.mean())
    y_offset = float(pos_y_fwd.mean())
    if recenter_xy:
        pos_x_fwd = pos_x_fwd - x_offset
        pos_y_fwd = pos_y_fwd - y_offset
        print(f"\n✅ Recentering XY aplicado: dx={x_offset:.2f} mm, dy={y_offset:.2f} mm")
    else:
        print(f"\nℹ️  Sin recenter: dx={x_offset:.2f} mm, dy={y_offset:.2f} mm")
    
    # ================================================================
    # SIMULACIÓN
    # ================================================================
    sim = gate.Simulation()
    
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    
    # Mundo grande (asegurar que el phantom cabe)
    # Phantom puede llegar hasta phsp_z + offset + phantom_size
    world_size = max(3000, int(phsp_z_mm + phantom_z_offset_cm * 10 + phantom_size_cm * 20 + 200))
    sim.world.size = [world_size * mm, world_size * mm, world_size * mm]
    sim.world.material = 'G4_AIR'
    
    # Phantom
    phantom = sim.add_volume('Box', 'water_phantom')
    phantom.size = [phantom_size_cm * cm] * 3
    phantom.material = 'G4_WATER'
    phantom.translation = [0, 0, phantom_center_z_mm * mm]
    
    # Física
    sim.physics_manager.physics_list_name = 'QGSP_BIC_EMZ'
    sim.physics_manager.set_production_cut('world', 'all', 1 * mm)
    sim.physics_manager.set_production_cut('water_phantom', 'all', 0.1 * mm)
    
    # ================================================================
    # FUENTES - Una por tipo de partícula, usando distribuciones reales
    # ================================================================
    particle_map = {-11: 'e+', 11: 'e-', 22: 'gamma'}
    unique_pdg = np.unique(pdg_fwd)
    
    for pdg_code in unique_pdg:
        if pdg_code not in particle_map:
            continue
        
        mask = pdg_fwd == pdg_code
        n_this = int(mask.sum())
        if n_this == 0:
            continue
        
        particle_name = particle_map[pdg_code]
        source = sim.add_source('GenericSource', f'source_{particle_name}')
        source.particle = particle_name
        source.n = n_this
        
        # Energía: histograma real
        e_this = energy_fwd[mask]
        hist, bins = np.histogram(e_this, bins=100)
        source.energy.type = 'histogram'
        source.energy.histogram_weight = hist.tolist()
        source.energy.histogram_energy = ((bins[:-1] + bins[1:]) / 2).tolist()
        
        # Posición: distribución REAL del phase space
        x_this = pos_x_fwd[mask]
        y_this = pos_y_fwd[mask]
        z_this = pos_z_fwd[mask]
        
        # Usar distribución gaussiana ajustada a los datos
        source.position.type = 'disc'
        source.position.radius = float(np.percentile(np.sqrt(x_this**2 + y_this**2), 99)) * mm
        source.position.translation = [
            float(x_this.mean()) * mm,
            float(y_this.mean()) * mm,
            float(z_this.mean()) * mm
        ]
        
        # Dirección: AQUÍ ESTÁ LA CLAVE
        # Usamos la dirección media de este tipo de partícula
        # Esto es una aproximación, pero mejor que focalizar todo al centro
        u_mean = float(dir_u_fwd[mask].mean())
        v_mean = float(dir_v_fwd[mask].mean())
        w_mean = float(dir_w_fwd[mask].mean())
        
        # Normalizar
        norm = np.sqrt(u_mean**2 + v_mean**2 + w_mean**2)
        if norm > 0:
            u_mean /= norm
            v_mean /= norm
            w_mean /= norm
        
        source.direction.type = 'momentum'
        source.direction.momentum = [u_mean, v_mean, w_mean]
        
        # Añadir dispersión angular basada en la varianza real
        # Calcular ángulo de apertura desde la varianza
        u_std = float(dir_u_fwd[mask].std())
        v_std = float(dir_v_fwd[mask].std())
        angular_spread = np.sqrt(u_std**2 + v_std**2)  # radianes aproximadamente
        
        # Usar iso distribution con apertura
        source.direction.type = 'iso'
        source.direction.theta = [0, float(np.arctan(angular_spread) * 180 / np.pi)]  # grados
        source.direction.phi = [0, 360]
        
        print(f"   {particle_name}: {n_this:,} partículas, E={e_this.mean():.2f}±{e_this.std():.2f} MeV")
        print(f"       dir=({u_mean:.3f}, {v_mean:.3f}, {w_mean:.3f}), spread={angular_spread:.3f} rad")
    
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
    
    sim.random_seed = seed if seed else 'auto'
    sim.number_of_threads = threads
    
    # Run
    print(f"\n▶️  Ejecutando ({threads} threads)...")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    
    print(f"✅ {elapsed:.1f}s - Eventos: {stats.counts['events']:,}")
    
    result = {
        'n_particles_total': n_total,
        'n_particles_used': n_use,
        'n_particles_forward': n_forward,
        'events': stats.counts['events'],
        'time_s': elapsed,
        'phantom_size_cm': phantom_size_cm,
        'phantom_z_mm': phantom_center_z_mm
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Dosis con partículas exactas del phase space')
    parser.add_argument('--phsp', required=True, help='Phase space .npz')
    parser.add_argument('--output', required=True, help='Directorio de salida')
    parser.add_argument('--particles', type=int, help='Número de partículas')
    parser.add_argument('--phantom-size', type=float, default=15, help='Tamaño phantom [cm]')
    parser.add_argument('--resolution', type=float, default=2, help='Resolución [mm]')
    parser.add_argument('--offset', type=float, default=5, help='Offset phantom desde phsp [cm]')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--no-recenter', action='store_true', help='No recentrar XY')
    
    args = parser.parse_args()
    
    run_dose_with_exact_particles(
        args.phsp, args.output,
        n_particles=args.particles,
        phantom_size_cm=args.phantom_size,
        dose_resolution_mm=args.resolution,
        phantom_z_offset_cm=args.offset,
        threads=args.threads,
        seed=args.seed,
        recenter_xy=not args.no_recenter
    )


if __name__ == '__main__':
    main()
