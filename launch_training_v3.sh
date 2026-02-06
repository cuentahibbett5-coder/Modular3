#!/bin/bash
# launch_training_v3.sh
# Script para entrenar DeepMC v3 en el cluster Yuca

set -e

echo "=========================================="
echo "DeepMC v3 Training Launch"
echo "=========================================="

cd /home/fer/fer/Modular3

# Verificar ambiente
echo "✓ Checking environment..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verificar dataset
echo ""
echo "✓ Checking dataset..."
if [ -d "dataset_pilot" ]; then
    echo "  ✓ dataset_pilot found"
else
    echo "  ✗ dataset_pilot NOT FOUND"
    echo "  Run create_dataset_pilot.py first!"
    exit 1
fi

# Crear directorio de resultados
mkdir -p runs/denoising_deepmc_v3
echo "  ✓ Results directory ready"

# Mostrar configuración
echo ""
echo "Configuration:"
echo "  Batch Size: 2"
echo "  Epochs: 100 (with early stopping @ patience=20)"
echo "  Learning Rate: 5e-4"
echo "  Loss Function: Exponential Weighted"
echo "  Architecture: DeepMC with SE Blocks + Residual"
echo "  Dual Input: Disabled (CT not available in pilot)"
echo ""

# Opciones
echo "Ready to train. Estimated time: 2.5-3.5 hours"
echo ""
echo "Launch options:"
echo "  1) Start training (default: nohup)"
echo "  2) Start training (foreground, interactive)"
echo "  3) Cancel"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Starting training in background..."
        nohup python train_deepmc_v3.py > training_deepmc_v3.log 2>&1 &
        echo "✓ Training started (PID: $!)"
        echo "  Monitor with: tail -f training_deepmc_v3.log"
        echo "  Check status: ps aux | grep train_deepmc_v3.py"
        ;;
    2)
        echo ""
        echo "Starting training in foreground..."
        python train_deepmc_v3.py
        ;;
    3)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Once training completes:"
echo "  python evaluate_deepmc_v3.py"
echo "=========================================="
