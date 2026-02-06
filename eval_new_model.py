#!/usr/bin/env python3
"""
Evaluación simple del modelo reentrenado.
Calcula PSNR, correlación y MSE en el conjunto de validación.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import json

# Desactivar MIOpen
torch.backends.cudnn.enabled = False

# ---- Configuración ----
MODEL_PATH = Path("runs/denoising_v2/best.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
PATCH_SIZE = (64, 64, 64)
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


def psnr(y_true, y_pred):
    """PSNR en dB."""
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def correlation(y_true, y_pred):
    """Correlación de Pearson."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return np.corrcoef(y_true_f, y_pred_f)[0, 1]


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
    print("EVALUACIÓN DEL MODELO REENTRENADO")
    print("=" * 70)
    
    # Cargar modelo
    assert MODEL_PATH.exists(), f"❌ No existe: {MODEL_PATH}"
    model = UNet3D(base_ch=32).to(DEVICE)
    ckpt = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ Modelo cargado de {MODEL_PATH}")
    print(f"   Epoch: {ckpt['epoch']}, Val loss: {ckpt['val_loss']:.6f}\n")
    
    # Recolectar pares val
    pairs = []
    n_targets = len(list(DATASET_ROOT.glob("target_*")))
    
    for pair_dir in sorted(VAL_DIR.glob("pair_*")):
        pair_num = int(pair_dir.name.split("_")[-1])
        target_idx = ((pair_num - 1) % n_targets) + 1
        target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
        
        if not target_mhd.exists():
            continue
        
        for level in INPUT_LEVELS:
            input_mhd = pair_dir / f"{level}.mhd"
            if not input_mhd.exists():
                input_mhd = pair_dir / level / "dose_edep.mhd"
            
            if input_mhd.exists():
                pairs.append((input_mhd, target_mhd, pair_dir.name, level))
    
    if not pairs:
        print("❌ No hay pares de validación")
        return
    
    print(f"📊 Evaluando {len(pairs)} pares...")
    
    # Evaluar
    all_psnr = []
    all_corr = []
    all_mse = []
    
    for input_mhd, target_mhd, pair_name, level in tqdm(pairs):
        try:
            inp = read_volume(input_mhd)
            tgt = read_volume(target_mhd)
            
            # Normalizar
            max_val = float(np.max(tgt))
            if max_val > 0:
                inp_norm = inp / max_val
                tgt_norm = tgt / max_val
            else:
                continue
            
            # Crop
            inp_crop = center_crop(inp_norm, PATCH_SIZE)
            tgt_crop = center_crop(tgt_norm, PATCH_SIZE)
            
            # A tensor
            inp_t = torch.from_numpy(inp_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
            tgt_t = torch.from_numpy(tgt_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Inferencia
            with torch.no_grad():
                pred_t = model(inp_t)
            
            pred = pred_t.squeeze().cpu().numpy()
            tgt_np = tgt_crop
            
            # Clip a [0, 1]
            pred = np.clip(pred, 0, 1)
            
            # Métricas
            p = psnr(tgt_np, pred)
            c = correlation(tgt_np, pred)
            m = np.mean((tgt_np - pred) ** 2)
            
            all_psnr.append(p)
            all_corr.append(c)
            all_mse.append(m)
            
        except Exception as e:
            print(f"  ⚠️  Error en {pair_name}/{level}: {e}")
    
    # Resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)
    print(f"PSNR:        {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"Correlación: {np.mean(all_corr):.4f} ± {np.std(all_corr):.4f}")
    print(f"MSE:         {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
    
    # Veredicto
    psnr_mean = np.mean(all_psnr)
    corr_mean = np.mean(all_corr)
    
    print("\n" + "=" * 70)
    if psnr_mean > 20 and corr_mean > 0.95:
        print("✅ EXCELENTE: El modelo funciona muy bien")
    elif psnr_mean > 15 and corr_mean > 0.85:
        print("✅ BUENO: El modelo funciona bien")
    elif psnr_mean > 10 and corr_mean > 0.70:
        print("⚠️  ACEPTABLE: El modelo funciona pero puede mejorar")
    else:
        print("❌ POBRE: El modelo necesita mejoras")
    print("=" * 70)
    
    # Guardar
    results = {
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "corr_mean": float(np.mean(all_corr)),
        "corr_std": float(np.std(all_corr)),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "n_samples": len(all_psnr)
    }
    
    with open("eval_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Resultados guardados en: eval_results_v2.json")


if __name__ == "__main__":
    main()
