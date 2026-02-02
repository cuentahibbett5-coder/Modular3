#!/bin/bash
# Configuración del X Server para visualización Qt en WSL
# 
# Uso: source setup_x_server.sh

echo "🖥️  Configurando X Server para WSL..."

# Detectar si estamos en WSL2
if grep -qi microsoft /proc/version; then
    echo "✓ WSL detectado"
    
    # Obtener IP del host Windows
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
    export LIBGL_ALWAYS_INDIRECT=1
    
    echo "✓ Variables configuradas:"
    echo "  DISPLAY=$DISPLAY"
    echo "  LIBGL_ALWAYS_INDIRECT=$LIBGL_ALWAYS_INDIRECT"
    
    echo ""
    echo "📋 Instrucciones para completar la configuración:"
    echo ""
    echo "1. Descargar e instalar VcXsrv en Windows:"
    echo "   https://sourceforge.net/projects/vcxsrv/"
    echo ""
    echo "2. Ejecutar XLaunch (buscar en menú inicio)"
    echo "   - Seleccionar: 'Multiple windows'"
    echo "   - Display number: 0"
    echo "   - Start no client"
    echo "   - ✓ Marcar 'Disable access control'"
    echo "   - Finish"
    echo ""
    echo "3. Agregar regla de firewall en Windows (PowerShell como Admin):"
    echo "   New-NetFirewallRule -DisplayName 'WSL VcXsrv' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 6000"
    echo ""
    echo "4. Probar la conexión:"
    echo "   xclock  # Si no está instalado: sudo apt install x11-apps"
    echo ""
    echo "5. Una vez funcionando, ejecutar la GUI del linac:"
    echo "   .venv/bin/python simulations/visualize_linac.py --visu qt"
    echo ""
    
    # Agregar a .bashrc si no está
    if ! grep -q "DISPLAY.*resolv.conf" ~/.bashrc; then
        echo ""
        echo "💾 ¿Agregar configuración al .bashrc? (s/n)"
        read -r response
        if [[ "$response" =~ ^[Ss]$ ]]; then
            echo "" >> ~/.bashrc
            echo "# X Server para WSL (GUI Geant4)" >> ~/.bashrc
            echo "export DISPLAY=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}'):0" >> ~/.bashrc
            echo "export LIBGL_ALWAYS_INDIRECT=1" >> ~/.bashrc
            echo "✓ Configuración agregada a ~/.bashrc"
        fi
    fi
else
    echo "ℹ️  No se detectó WSL, puede que ya tengas DISPLAY configurado"
    echo "  DISPLAY actual: $DISPLAY"
fi

echo ""
echo "✅ Configuración completada para esta sesión"
echo "   Para hacer permanente, agregar al ~/.bashrc"
