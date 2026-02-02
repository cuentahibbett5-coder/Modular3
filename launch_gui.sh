#!/bin/bash
# Script para probar y lanzar la GUI OpenGL de Geant4
# Uso: ./launch_gui.sh

echo "🔍 Verificando configuración del X Server..."

# Configurar variables
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1

echo "✓ Variables configuradas:"
echo "  DISPLAY=$DISPLAY"
echo ""

# Probar conexión
echo "Probando conexión con X Server..."
if command -v xdpyinfo &> /dev/null; then
    if timeout 2 xdpyinfo &> /dev/null; then
        echo "✅ Conexión con X Server exitosa!"
        echo ""
        echo "🚀 Lanzando GUI del Linac..."
        .venv/bin/python simulations/visualize_linac.py --visu qt
    else
        echo "❌ No se pudo conectar con X Server"
        echo ""
        echo "Verifica que:"
        echo "1. VcXsrv está corriendo (busca icono X en bandeja de Windows)"
        echo "2. Configuraste 'Disable access control' en XLaunch"
        echo "3. No hay firewall bloqueando el puerto 6000"
        echo ""
        echo "Para instalar VcXsrv:"
        echo "  https://sourceforge.net/projects/vcxsrv/"
    fi
else
    echo "⚠️  xdpyinfo no está instalado, intentando de todos modos..."
    echo ""
    echo "🚀 Lanzando GUI del Linac..."
    .venv/bin/python simulations/visualize_linac.py --visu qt
fi
