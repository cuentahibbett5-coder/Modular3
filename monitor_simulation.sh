#!/bin/bash

# Monitor de simulación - muestra progreso y estimaciones

echo "=================================================="
echo "Monitor de Simulaciones Modular3"
echo "=================================================="

# Mostrar procesos activos
echo ""
echo "📊 Procesos Python/OpenGate activos:"
ps aux | grep -E "python.*dose_simulation|GATE|geant4" | grep -v grep || echo "  (ninguno)"

# Mostrar outputs disponibles
echo ""
echo "📁 Datasets generados:"
ls -lhd /home/fer/fer/Modular3/output/dose_maps/*/ 2>/dev/null | awk '{print "  " $9 " - " $5}'

# Mostrar logs recientes
echo ""
echo "📋 Últimos logs:"
for logfile in /home/fer/fer/Modular3/output/*.log; do
    if [ -f "$logfile" ]; then
        echo "  $(basename $logfile):"
        tail -3 "$logfile" | sed 's/^/    /'
    fi
done

echo ""
echo "=================================================="
