# Visualización del Linac - Guía de Uso

## Archivos generados

### GDML (Geometry Description Markup Language)
- **Archivo**: `output/linac_geometry.gdml`
- **Uso**: Formato estándar de Geant4 para describir geometrías

## Opciones de visualización

### 1. ROOT (Recomendado para análisis)
Si tienes ROOT instalado:
```bash
root
root [0] TGeoManager::Import("output/linac_geometry.gdml")
root [1] gGeoManager->GetTopVolume()->Draw("ogl")
```

### 2. FreeCAD
- Instalar FreeCAD: `sudo apt install freecad`
- Abrir archivo GDML directamente o convertir a STEP

### 3. Visualización Qt con X Server (WSL)

#### Opción A: Usar VcXsrv (Windows)
1. Instalar VcXsrv en Windows
2. Ejecutar XLaunch con "Disable access control"
3. En WSL:
```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
.venv/bin/python simulations/visualize_linac.py --linac versa --visu qt
```

#### Opción B: Usar WSLg (Windows 11)
Si tienes Windows 11 con WSLg ya configurado:
```bash
.venv/bin/python simulations/visualize_linac.py --linac versa --visu qt
```

### 4. Análisis de Phase Space con matplotlib

Visualización de los datos de simulación sin necesidad de Qt:

```python
import uproot
import matplotlib.pyplot as plt
import numpy as np

# Leer datos
f = uproot.open("data/phase_space/versa_6mv_1e6.root")
tree = f["versa_linac_phsp_plane_phsp"]
data = tree.arrays()

# Visualizar distribución espacial
plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.scatter(data['PrePosition_X'], data['PrePosition_Y'], 
           c=data['KineticEnergy'], s=1, alpha=0.5)
plt.colorbar(label='Energy (MeV)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title('Phase Space - Distribución Espacial')

plt.subplot(132)
plt.hist(data['KineticEnergy'], bins=100)
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.title('Espectro de Energía')

plt.subplot(133)
pdg_codes = data['PDGCode']
unique, counts = np.unique(pdg_codes, return_counts=True)
particles = {22: 'Fotones', 11: 'Electrones', -11: 'Positrones'}
labels = [particles.get(p, f'PDG {p}') for p in unique]
plt.bar(labels, counts)
plt.ylabel('Counts')
plt.title('Tipos de Partículas')

plt.tight_layout()
plt.savefig('output/phase_space_analysis.png', dpi=150)
plt.show()
```

## Visualización web con Jupyter

Para visualización interactiva en navegador:

```bash
.venv/bin/jupyter notebook
```

Luego crear un notebook con plotly para visualización 3D interactiva.

## Comando rápido para verificar geometría

```bash
# Exportar a GDML
.venv/bin/python simulations/visualize_linac.py --visu gdml

# Ver estructura del archivo
grep "volume name" output/linac_geometry.gdml | head -20
```

## Archivos disponibles

- `output/linac_geometry.gdml` - Geometría completa del linac
- `data/phase_space/versa_6mv_1e6.root` - Datos de simulación
- `simulations/visualize_linac.py` - Script de visualización
