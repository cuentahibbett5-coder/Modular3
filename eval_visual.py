#!/usr/bin/env python3
"""
Visualización de una predicción del modelo.
Grafica slices axial, sagital y coronal para verificar si el modelo
realmente aprendió la geometría del haz de radiación.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

# Desactivar MIOpen
torch.backends.cudnn.enabled = False

# ---- Configuración ----
MODEL_PATH = Path("runs/denoising_v2/best.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
PATCH_SIZE = (128, 128, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")


# ---- Funciones ----
def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee archivo .mhd."""
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    img = sitk.ReadImage(str(mhd_path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def center_crop(vol, patch_size):
    """Crop central."""
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


# ---- Modelo UNet3D ----
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)
        self.out = nn.Conv3d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# ---- Main ----
def main():
    print("=" * 70)
    print("VISUALIZACIÓN DE PREDICCIÓN")
    print("=" * 70)
    
    # Cargar modelo
    assert MODEL_PATH.exists(), f"❌ No existe: {MODEL_PATH}"
    model = UNet3D(base_ch=32).to(DEVICE)
    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Modelo cargado")
    
    # Buscar un par de validación
    pair_021 = VAL_DIR / "pair_021"
    if not pair_021.exists():
        print(f"❌ No existe {pair_021}")
        return
    
    # Input: pair_021/input_1M.mhd
    input_mhd = pair_021 / "input_1M.mhd"
    if not input_mhd.exists():
        input_mhd = pair_021 / "input_1M" / "dose_edep.mhd"
    
    if not input_mhd.exists():
        print(f"❌ No existe input: {input_mhd}")
        return
    
    # Target: target_1/dose_edep.mhd (pair 021 usa target (21-1)%5+1 = 1)
    target_mhd = DATASET_ROOT / "target_1" / "dose_edep.mhd"
    if not target_mhd.exists():
        print(f"❌ No existe target: {target_mhd}")
        return
    
    print(f"\n📂 Cargando datos...")
    print(f"   Input:  {input_mhd}")
    print(f"   Target: {target_mhd}")
    
    # Leer volúmenes
    inp = read_volume(input_mhd)
    tgt = read_volume(target_mhd)
    
    print(f"   Input shape:  {inp.shape}")
    print(f"   Target shape: {tgt.shape}")
    
    # Normalizar (MISMO que en train)
    max_val = float(np.max(tgt))
    if max_val <= 0:
        print("❌ Target es todo ceros")
        return
    
    inp_norm = inp / max_val
    tgt_norm = tgt / max_val
    
    print(f"   Input range:  [{inp_norm.min():.6f}, {inp_norm.max():.6f}]")
    print(f"   Target range: [{tgt_norm.min():.6f}, {tgt_norm.max():.6f}]")
    
    # Crop (MISMO que en train)
    inp_crop = center_crop(inp_norm, PATCH_SIZE)
    tgt_crop = center_crop(tgt_norm, PATCH_SIZE)
    
    print(f"   Cropped shape: {inp_crop.shape}")
    
    # Tensor
    inp_t = torch.from_numpy(inp_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
    tgt_t = torch.from_numpy(tgt_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Inferencia
    print(f"\n🧠 Inferencia...")
    with torch.no_grad():
        pred_t = model(inp_t)
    
    pred = pred_t.squeeze().cpu().numpy()
    pred = np.clip(pred, 0, 1)
    
    print(f"   Predicción shape: {pred.shape}")
    print(f"   Predicción range: [{pred.min():.6f}, {pred.max():.6f}]")
    
    # ---- Visualización ----
    print(f"\n📊 Generando visualización...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Denoising MC: Input (1M) → Target (29.4M clean) vs Prediction",
        fontsize=16, fontweight='bold'
    )
    
    # Slices: axial (Z), sagital (Y), coronal (X)
    slices = {
        'Axial (Z)': PATCH_SIZE[0] // 2,
        'Sagital (Y)': PATCH_SIZE[1] // 2,
        'Coronal (X)': PATCH_SIZE[2] // 2,
    }
    
    slice_names = list(slices.keys())
    cmap = 'hot'
    
    for col, (slice_name, idx) in enumerate(slices.items()):
        # Input
        if slice_name == 'Axial (Z)':
            inp_slice = inp_crop[idx, :, :]
            tgt_slice = tgt_crop[idx, :, :]
            pred_slice = pred[idx, :, :]
        elif slice_name == 'Sagital (Y)':
            inp_slice = inp_crop[:, idx, :]
            tgt_slice = tgt_crop[:, idx, :]
            pred_slice = pred[:, idx, :]
        else:  # Coronal (X)
            inp_slice = inp_crop[:, :, idx]
            tgt_slice = tgt_crop[:, :, idx]
            pred_slice = pred[:, :, idx]
        
        # Fila 0: Input
        ax = axes[0, col]
        im = ax.imshow(inp_slice, cmap=cmap, origin='lower')
        ax.set_title(f'{slice_name}\n(Input 1M)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Fila 1: Target
        ax = axes[1, col]
        im = ax.imshow(tgt_slice, cmap=cmap, origin='lower')
        ax.set_title(f'{slice_name}\n(Target 29.4M)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Fila 2: Predicción
        ax = axes[2, col]
        im = ax.imshow(pred_slice, cmap=cmap, origin='lower')
        ax.set_title(f'{slice_name}\n(Predicción)', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("eval_visual.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Guardado: eval_visual.png")
    
    # ---- Análisis de calidad ----
    print(f"\n📋 Análisis:")
    print(f"   - ¿Se ve la forma del haz? (target y predicción deberían ser similares)")
    print(f"   - ¿Predicción es ruido? (valores aleatorios)")
    print(f"   - ¿Predicción es plana? (modelo no aprendió nada)")
    
    # Diferencia
    diff = np.abs(tgt_crop - pred)
    mae = np.mean(diff)
    
    print(f"\n   MAE (Error Absoluto Medio): {mae:.6f}")
    print(f"   (Más bajo = mejor. Debería ser < 0.01 si funciona bien)")
    
    if mae < 0.01:
        print(f"\n✅ ¡El modelo parece funcionar bien!")
    elif mae < 0.05:
        print(f"\n⚠️  El modelo funciona pero necesita mejoras")
    else:
        print(f"\n❌ El modelo no está aprendiendo correctamente")


if __name__ == "__main__":
    main()
