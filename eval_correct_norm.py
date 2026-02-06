#!/usr/bin/env python3
"""
Evaluación de todos los niveles con normalización CORRECTA (por max input).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.backends.cudnn.enabled = False

# ---- Configuración ----
MODEL_PATH = Path("runs/denoising_v2/best.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
PATCH_SIZE = (64, 64, 64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}\n")


def read_volume(mhd_path: Path) -> np.ndarray:
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    img = sitk.ReadImage(str(mhd_path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def center_crop(vol, patch_size):
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


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


print("=" * 70)
print("EVALUACIÓN CON NORMALIZACIÓN CORRECTA (por max input)")
print("=" * 70)

# Cargar modelo
model = UNet3D(base_ch=32).to(DEVICE)
ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"✅ Modelo cargado\n")

# Recolectar datos
results = {}
n_targets = len(list(DATASET_ROOT.glob("target_*")))

for pair_dir in tqdm(sorted(VAL_DIR.glob("pair_*")), desc="Procesando pairs"):
    pair_num = int(pair_dir.name.split("_")[-1])
    target_idx = ((pair_num - 1) % n_targets) + 1
    target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
    
    if not target_mhd.exists():
        continue
    
    tgt = read_volume(target_mhd)
    
    for level in INPUT_LEVELS:
        input_mhd = pair_dir / f"{level}.mhd"
        if not input_mhd.exists():
            input_mhd = pair_dir / level / "dose_edep.mhd"
        
        if not input_mhd.exists():
            continue
        
        inp = read_volume(input_mhd)
        
        # 🔑 Normalizar por MAX(INPUT) - MISMO que en training
        max_inp = float(np.max(inp))
        if max_inp > 0:
            inp_norm = inp / max_inp
            tgt_norm = tgt / max_inp
        else:
            continue
        
        # Crop
        inp_crop = center_crop(inp_norm, PATCH_SIZE)
        tgt_crop = center_crop(tgt_norm, PATCH_SIZE)
        
        # Inferencia
        inp_t = torch.from_numpy(inp_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_t = model(inp_t)
        pred = pred_t.squeeze().cpu().numpy()
        pred = np.clip(pred, 0, np.inf)  # Sin cap en max, permite escalar
        
        # Métricas
        mae = np.mean(np.abs(tgt_crop - pred))
        
        if level not in results:
            results[level] = {'mae': [], 'inp_max': [], 'pred_max': [], 'tgt_max': []}
        
        results[level]['mae'].append(mae)
        results[level]['inp_max'].append(np.max(inp_norm))
        results[level]['pred_max'].append(np.max(pred))
        results[level]['tgt_max'].append(np.max(tgt_norm))

# Imprimir resultados
print("\n" + "=" * 100)
print("📊 RESULTADOS (normalización por MAX INPUT)")
print("=" * 100)
print(f"{'Nivel':<12} {'Input Max':<12} {'Target Max':<12} {'Pred Max':<12} {'MAE':<12} {'Pred/Tgt':<12}")
print("-" * 100)

for level in INPUT_LEVELS:
    if level not in results:
        continue
    
    inp_max_m = np.mean(results[level]['inp_max'])
    tgt_max_m = np.mean(results[level]['tgt_max'])
    pred_max_m = np.mean(results[level]['pred_max'])
    mae_m = np.mean(results[level]['mae'])
    ratio = pred_max_m / tgt_max_m if tgt_max_m > 0 else 0
    
    print(f"{level:<12} {inp_max_m:<12.4f} {tgt_max_m:<12.4f} {pred_max_m:<12.4f} {mae_m:<12.6f} {ratio:<12.2f}x")

print("=" * 100)
print("\n📋 ANÁLISIS:")
print("   - Pred/Tgt ratio cercano a 1.0 = modelo escalando correctamente")
print("   - Si ratio < 1.0: modelo sigue sub-prediciendo")
print("   - Si ratio > 1.0: modelo sobre-amplifica")
