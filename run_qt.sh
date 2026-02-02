#!/bin/bash
# Script para ejecutar aplicaciones Qt con OpenGate
# Resuelve el conflicto de símbolos entre Qt del sistema y PyQt5

cd /home/fer/fer/Modular3

# Activar entorno virtual
source .venv/bin/activate

# Configurar librería Qt de PyQt5
export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH"

# Forzar xcb en lugar de wayland
export QT_QPA_PLATFORM=xcb

# Ejecutar el script de visualización
python simulations/visualize_linac.py "$@"
