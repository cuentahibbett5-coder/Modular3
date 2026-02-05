#!/usr/bin/env python3
"""
Test simple para identificar qué causa el segfault.
"""

import sys
from pathlib import Path

print("1. Importando torch...")
import torch
print(f"   ✓ PyTorch {torch.__version__}")

print("2. Verificando CUDA...")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

print("3. Importando matplotlib...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("   ✓ Matplotlib OK")

print("4. Importando numpy...")
import numpy as np
print("   ✓ NumPy OK")

print("5. Importando SimpleITK...")
try:
    import SimpleITK as sitk
    print("   ✓ SimpleITK OK")
except Exception as e:
    print(f"   ✗ Error con SimpleITK: {e}")
    sys.exit(1)

print("6. Cargando modelo...")
try:
    from training.model import UNet3D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    print(f"   ✓ Modelo creado en {device}")
    
    checkpoint = Path("runs/denoising/best.pt")
    if checkpoint.exists():
        ckpt = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(ckpt["model"])
        print("   ✓ Checkpoint cargado")
    else:
        print(f"   ⚠ Checkpoint no encontrado: {checkpoint}")
except Exception as e:
    print(f"   ✗ Error con modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("7. Probando dataset...")
try:
    from training.dataset import DosePairDataset
    val_ds = DosePairDataset(
        root_dir="dataset_pilot",
        split="val",
        patch_size=(64, 64, 64),
        cache_size=0,
        normalize=True,
        seed=4321,
    )
    print(f"   ✓ Dataset creado: {len(val_ds)} samples")
    
    print("8. Cargando primer sample...")
    batch = val_ds[0]
    print(f"   ✓ Sample cargado - input shape: {batch['input'].shape}, target shape: {batch['target'].shape}")
    
except Exception as e:
    print(f"   ✗ Error con dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ TODAS LAS PRUEBAS PASARON")
print("El problema puede estar en el loop de inferencia o visualización.")
