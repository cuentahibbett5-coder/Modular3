"""
Simulación de Acelerador Lineal con GATE 10 / OpenGate

Usa linacs pre-configurados de OpenGate (contrib.linacs) en lugar de
construir geometría desde cero. Más realista y validado.

Linacs soportados:
- Elekta Versa (opengate.contrib.linacs.elektaversa)
- Varian TrueBeam (opengate.contrib.linacs.varian)

Autor: Proyecto Modular 3 - CUCEI
Fecha: Febrero 2026
"""

import numpy as np
from pathlib import Path
import opengate as gate

# Intentar importar linacs pre-configurados
try:
    import opengate.contrib.linacs.elektaversa as versa
    VERSA_AVAILABLE = True
except ImportError:
    VERSA_AVAILABLE = False
    print("⚠️  Elekta Versa no disponible")

try:
    import opengate.contrib.linacs.varian as varian
    VARIAN_AVAILABLE = True
except ImportError:
    VARIAN_AVAILABLE = False
    print("⚠️  Varian no disponible")


class LinacSimulation:
    """
    Simulación de acelerador lineal usando modelos pre-configurados de OpenGate.
    """
    
    def __init__(self, linac_type='versa', energy='6MV', field_size=(10, 10)):
        """
        Inicializa la simulación del Linac.
        
        Args:
            linac_type: 'versa' o 'varian'
            energy: '6MV', '10MV', '15MV', '18MV'
            field_size: Tamaño del campo en cm (x, y)
        """
        self.linac_type = linac_type.lower()
        self.energy = energy
        self.field_size = field_size
        
        # Crear simulación
        self.sim = gate.Simulation()
        
        # Referencias
        self.linac = None
        self.source = None
        self.phantom = None
        
        print(f"✅ LinacSimulation: {linac_type.upper()} {energy}, campo {field_size[0]}×{field_size[1]} cm²")
    
    def setup_linac(self):
        """Configura el linac usando modelos pre-construidos."""
        
        if self.linac_type == 'versa' and VERSA_AVAILABLE:
            print("📦 Usando Elekta Versa (geometría validada)")
            linac_name = f"{self.linac_type}_linac"
            self.linac = versa.add_linac(self.sim, linac_name=linac_name)
            ekin = self._energy_to_ekin(self.energy)
            self.source = versa.add_electron_source(self.sim, linac_name, ekin, sx=1.0, sy=1.0)
            
        elif self.linac_type == 'varian' and VARIAN_AVAILABLE:
            print("📦 Usando Varian TrueBeam")
            self.linac = varian.add_linac(self.sim)
            varian.set_default_source(self.sim, self.linac, self.energy)
            
        else:
            # Fallback: geometría mínima
            print("⚠️  Linac pre-configurado no disponible")
            print("   Usando geometría simplificada (solo target)")
            self._setup_minimal_linac()
        
        # Configurar jaws para el campo deseado
        self._set_field_size(self.field_size)
        
        return self.linac
    
    def _setup_minimal_linac(self):
        """Geometría mínima si no hay linacs pre-configurados."""
        mm = gate.g4_units.mm
        MeV = gate.g4_units.MeV
        
        # Contenedor del cabezal
        self.linac = self.sim.add_volume('Box', 'linac_head')
        self.linac.size = [400 * mm, 400 * mm, 500 * mm]
        self.linac.material = "G4_AIR"
        self.linac.translation = [0, 0, -150 * mm]
        
        # Target de tungsteno (lo mínimo indispensable)
        target = self.sim.add_volume('Tubs', 'target')
        target.mother = self.linac.name
        target.rmin = 0
        target.rmax = 2.5 * mm
        target.dz = 0.75 * mm
        target.material = "G4_W"
        target.translation = [0, 0, 0]
        target.color = [0.8, 0.8, 0.2, 0.8]
        
        # Fuente de electrones básica
        self.source = self.sim.add_source('GenericSource', 'electron_source')
        self.source.particle = 'e-'
        self.source.energy.type = 'gauss'
        self.source.energy.mean = 5.8 * MeV
        self.source.energy.sigma = 0.17 * MeV  # 3% FWHM
        self.source.position.type = 'disc'
        self.source.position.radius = 3 * mm
        self.source.position.translation = [0, 0, -50 * mm]
        self.source.direction.type = 'focused'
        self.source.direction.focus_point = [0, 0, 0]
    
    def _energy_to_ekin(self, energy_label):
        """Convierte energía nominal (ej. 6MV) a energía cinética de electrones (MeV)."""
        mapping = {
            '6MV': 5.8,
            '10MV': 10.5,
            '15MV': 15.0,
            '18MV': 18.0,
        }
        if energy_label in mapping:
            return mapping[energy_label]
        try:
            return float(str(energy_label).upper().replace('MV', '').strip())
        except Exception:
            print(f"⚠️  Energía desconocida '{energy_label}', usando 6MV por defecto")
            return mapping['6MV']
    
    def _set_field_size(self, field_size):
        """Configura el tamaño del campo (si el linac lo permite)."""
        x_cm, y_cm = field_size
        
        # Intentar configurar jaws si el linac pre-configurado lo permite
        if hasattr(self.linac, 'set_jaws_opening'):
            self.linac.set_jaws_opening(x_cm, y_cm)
            print(f"  ✓ Campo configurado: {x_cm}×{y_cm} cm²")
        else:
            print(f"  ℹ️  Campo por defecto (jaws no configurables en este modo)")
    
    def add_phase_space_actor(self, output_path, plane_position=350):
        """
        Añade actor de phase space para guardar partículas.
        
        Args:
            output_path: Ruta del archivo .root de salida
            plane_position: Posición Z del plano en mm
        """
        mm = gate.g4_units.mm
        
        # Si usamos versa, usar sus funciones nativas
        if self.linac_type == 'versa' and VERSA_AVAILABLE:
            linac_name = f"{self.linac_type}_linac"
            ps_plane = versa.add_phase_space_plane(self.sim, linac_name, src_phsp_distance=plane_position)
            self.phase_space_actor = versa.add_phase_space_actor(self.sim, ps_plane.name)
            self.phase_space_actor.output_filename = str(output_path)
            print(f"✅ Phase Space Actor añadido (Versa nativo): {output_path}")
            print(f"   Plano en Z = {plane_position} mm")
            return self.phase_space_actor
        
        # Crear plano de phase space manualmente para otros linacs
        ps_plane = self.sim.add_volume('Tubs', 'phase_space_plane')
        ps_plane.rmin = 0
        ps_plane.rmax = 150 * mm
        ps_plane.dz = 1 * mm
        ps_plane.material = 'G4_AIR'
        ps_plane.translation = [0, 0, plane_position * mm]
        ps_plane.color = [0, 1, 0, 0.3]  # Verde transparente
        
        # Añadir actor
        self.phase_space_actor = self.sim.add_actor('PhaseSpaceActor', 'phase_space')
        self.phase_space_actor.attached_to = ps_plane.name
        self.phase_space_actor.output_filename = str(output_path)
        self.phase_space_actor.store_absorbed_event = True
        # Usar solo los atributos que funcionan con elektaversa
        self.phase_space_actor.attributes = [
            "KineticEnergy",
            "Weight",
            "PrePosition",
            "PrePositionLocal",
            "PreDirection",
            "PreDirectionLocal",
            "PDGCode",
        ]
        
        print(f"✅ Phase Space Actor añadido: {output_path}")
        print(f"   Plano en Z = {plane_position} mm")
        
        return self.phase_space_actor
    
    def add_water_phantom(self, size=(300, 300, 300), position=(0, 0, 1000)):
        """
        Añade fantoma de agua.
        
        Args:
            size: Tamaño (x, y, z) en mm
            position: Posición (x, y, z) en mm
        """
        mm = gate.g4_units.mm
        
        self.phantom = self.sim.add_volume('Box', 'water_phantom')
        self.phantom.size = [s * mm for s in size]
        self.phantom.material = 'G4_WATER'
        self.phantom.translation = [p * mm for p in position]
        self.phantom.color = [0, 0, 1, 0.3]  # Azul transparente
        
        print(f"✅ Fantoma de agua: {size[0]}×{size[1]}×{size[2]} mm³")
        print(f"   Posición: {position} mm (SAD = {position[2]} mm)")
        
        return self.phantom
    
    def add_dose_actor(self, output_path, size=(100, 100, 100), spacing=(3, 3, 3)):
        """
        Añade actor de dosis al fantoma.
        
        Args:
            output_path: Ruta del archivo de salida (.mhd)
            size: Número de voxels (nx, ny, nz)
            spacing: Espaciado de voxels en mm
        """
        if self.phantom is None:
            raise ValueError("Debe añadir un phantom primero con add_water_phantom()")
        
        mm = gate.g4_units.mm
        
        dose_actor = self.sim.add_actor('DoseActor', 'dose')
        dose_actor.attached_to = self.phantom.name
        dose_actor.output_filename = str(output_path)
        dose_actor.size = list(size)
        dose_actor.spacing = [s * mm for s in spacing]
        
        print(f"✅ Dose Actor añadido: {output_path}")
        print(f"   Voxels: {size[0]}×{size[1]}×{size[2]}")
        print(f"   Espaciado: {spacing[0]}×{spacing[1]}×{spacing[2]} mm")
        
        return dose_actor
    
    def setup_physics(self, physics_list='QGSP_BIC_EMZ'):
        """Configura la lista de física de Geant4."""
        self.sim.physics_manager.physics_list_name = physics_list
        
        # Cuts de producción (mm)
        mm = gate.g4_units.mm
        self.sim.physics_manager.global_production_cuts.all = 1 * mm
        
        # Cuts específicos en el fantoma
        if self.phantom:
            cuts = self.sim.physics_manager.add_production_cuts('phantom_cuts')
            cuts.volume = self.phantom.name
            cuts.gamma = 0.1 * mm
            cuts.electron = 0.1 * mm
            cuts.positron = 0.1 * mm
        
        print(f"✅ Física: {physics_list}")
        print(f"   Production cuts: 1.0 mm (global), 0.1 mm (phantom)")
    
    def add_progress_actor(self, report_every=100000):
        """Añade actor para reportar progreso cada N partículas."""
        stats = self.sim.add_actor('SimulationStatisticsActor', 'progress_stats')
        stats.track_types_flag = False
        stats.output_filename = ''  # No guardar archivo
        
        # Callback de progreso
        self._progress_interval = report_every
        self._last_report = 0
        
        print(f"✅ Reporte de progreso cada {report_every:,} partículas")
        return stats
    
    def run(self, n_particles=1e7, n_threads=None):
        """
        Ejecuta la simulación.
        
        Args:
            n_particles: Número de partículas a simular
            n_threads: Número de threads (None = automático basado en CPU)
        """
        import os
        
        # Configurar multi-threading
        if n_threads is None:
            n_threads = os.cpu_count() or 4
        self.sim.number_of_threads = n_threads
        
        # Configurar número de partículas
        if self.source:
            self.source.n = int(n_particles)
        else:
            # Si usamos linac pre-configurado, la fuente ya está configurada
            pass
        
        # Añadir actor de progreso
        self.add_progress_actor(report_every=100000)
        
        # Configurar verbose para ver progreso
        self.sim.g4_verbose_level = 0
        self.sim.progress_bar = True  # Barra de progreso nativa de OpenGate
        
        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO SIMULACIÓN")
        print(f"{'='*60}")
        print(f"Partículas: {n_particles:.0e}")
        print(f"Threads: {n_threads}")
        print(f"Linac: {self.linac_type.upper()} {self.energy}")
        print(f"Campo: {self.field_size[0]}×{self.field_size[1]} cm²")
        print(f"{'='*60}\n")
        
        # Ejecutar
        self.sim.run()
        
        print(f"\n{'='*60}")
        print(f"✅ SIMULACIÓN COMPLETADA")
        print(f"{'='*60}\n")
        
        return self.sim


def main():
    """Ejemplo de uso."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simular Linac con OpenGate')
    parser.add_argument('--linac', type=str, default='versa', 
                       choices=['versa', 'varian'],
                       help='Tipo de linac')
    parser.add_argument('--energy', type=str, default='6MV',
                       help='Energía nominal (6MV, 10MV, etc.)')
    parser.add_argument('--field', nargs=2, type=float, default=[10, 10],
                       help='Tamaño de campo en cm (x y)')
    parser.add_argument('--particles', type=float, default=1e7,
                       help='Número de partículas')
    parser.add_argument('--output', type=str, default='data/phase_space/linac_ps.root',
                       help='Archivo de salida')
    parser.add_argument('--threads', type=int, default=None,
                       help='Número de threads (None = auto-detect)')
    
    args = parser.parse_args()
    
    # Crear simulación
    linac_sim = LinacSimulation(
        linac_type=args.linac,
        energy=args.energy,
        field_size=tuple(args.field)
    )
    
    # Configurar
    linac_sim.setup_linac()
    linac_sim.setup_physics()
    linac_sim.add_phase_space_actor(args.output)
    
    # Ejecutar
    linac_sim.run(n_particles=args.particles, n_threads=args.threads)
    
    print(f"\n✅ Phase space guardado en: {args.output}")


if __name__ == '__main__':
    main()
