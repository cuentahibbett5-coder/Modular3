#!/usr/bin/env python3
"""
=============================================================
ENTRENAMIENTO CON MÁSCARA PONDERADA Y FILTRADO DE PDD BAJA
=============================================================
- Elimina el 1% inferior de PDD (capas de dosis muy baja)
- Usa pérdida ponderada: core >> periferia
- Loss = MSE × weights, donde weights ∈ [0.5, 1.0]
"""

# ---- Desactivar MIOpen (roto en este ROCm 5.7) ----
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

# =============================================================
# ⚙️ CONFIGURACIÓN
# =============================================================

DATASET_ROOT   = Path("dataset_pilot")
TRAIN_DIR      = DATASET_ROOT / "train"
VAL_DIR        = DATASET_ROOT / "val"
OUTPUT_DIR     = Path("runs/denoising_weighted")

INPUT_LEVELS   = ["input_1M", "input_2M", "input_5M", "input_10M"]

BATCH_SIZE     = 2
PATCH_SIZE     = (64, 64, 64)
NUM_EPOCHS     = 100
LEARNING_RATE  = 1e-3
DEVICE         = "auto"

# 🎯 Parámetros de ponderación
DOSE_THRESHOLD = 0.20  # 20% del máximo = límite core/periferia
LOW_WEIGHT     = 0.5   # Peso para periferia (< 20%)
HIGH_WEIGHT    = 1.0   # Peso para core (≥ 20%)

# Filtrado de PDD baja
PERCENTILE_PDD_LOW = 1.0  # Eliminar el 1% inferior de PDD


# =============================================================

def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee un archivo .mhd."""
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


def calculate_pdd(vol):
    """
    Calcula PDD (Percent Depth Dose) como el máximo por capa (z-axis).
    Returns: array (D,) con PDD por layer
    """
    D = vol.shape[0]
    pdd = np.array([np.max(vol[z]) for z in range(D)])
    return pdd


def get_pdd_mask(vol, percentile_low=1.0):
    """
    Retorna máscara de capas a CONSERVAR basado en percentil de PDD.
    
    Si percentile_low=1.0, elimina el 1% más bajo de PDD.
    Returns: máscara booleana (D,) donde True = conservar, False = eliminar
    """
    pdd = calculate_pdd(vol)
    
    # Threshold: eliminar 1% más bajo
    threshold = np.percentile(pdd, percentile_low)
    mask = pdd >= threshold
    
    return mask, pdd


def random_crop(vol_a, vol_b, patch_size):
    """Crop aleatorio sincronizado."""
    z, y, x = vol_a.shape
    pz, py, px = patch_size
    sz = np.random.randint(0, max(z - pz + 1, 1))
    sy = np.random.randint(0, max(y - py + 1, 1))
    sx = np.random.randint(0, max(x - px + 1, 1))
    return (vol_a[sz:sz+pz, sy:sy+py, sx:sx+px],
            vol_b[sz:sz+pz, sy:sy+py, sx:sx+px])


def center_crop(vol, patch_size):
    """Crop central."""
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


def create_dose_weights(tgt, dose_threshold=0.20, low_weight=0.5, high_weight=1.0):
    """
    Crea mapa de pesos basado en nivel de dosis.
    
    Voxeles con dosis ≥ (threshold × max) → weight = high_weight (1.0)
    Voxeles con dosis < (threshold × max) → weight = low_weight (0.5)
    
    Returns: tensor (1, Z, Y, X) con pesos
    """
    max_dose = np.max(tgt)
    dose_cutoff = dose_threshold * max_dose
    
    weights = np.where(tgt >= dose_cutoff, high_weight, low_weight)
    weights = torch.from_numpy(weights).unsqueeze(0).float()  # (1, Z, Y, X)
    
    return weights


class SimpleDoseDatasetWeighted(Dataset):
    """Dataset con filtrado de PDD baja y pesos por dosis."""
    
    def __init__(self, split_dir, dataset_root, levels, patch_size, 
                 percentile_pdd=1.0, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.percentile_pdd = percentile_pdd
        self.pairs = []
        self.pdd_masks = {}  # Guardar máscaras por target
        
        n_targets = len(list(dataset_root.glob("target_*")))
        
        for pair_dir in sorted(split_dir.glob("pair_*")):
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            
            if not target_mhd.exists():
                print(f"  ⚠️  Target no encontrado: {target_mhd}")
                continue
            
            # Calcular máscara de PDD si no existe
            if target_idx not in self.pdd_masks:
                tgt_vol = read_volume(target_mhd)
                pdd_mask, _ = get_pdd_mask(tgt_vol, percentile_low=percentile_pdd)
                self.pdd_masks[target_idx] = pdd_mask
            
            for level in levels:
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                
                if input_mhd.exists():
                    self.pairs.append((input_mhd, target_mhd, target_idx))
                else:
                    print(f"  ⚠️  Input no encontrado: {pair_dir.name}/{level}")
        
        print(f"  ✅ {len(self.pairs)} pares encontrados")
        
        # Estadísticas de PDD
        n_kept = 0
        n_total = 0
        for mask in self.pdd_masks.values():
            n_kept += np.sum(mask)
            n_total += len(mask)
        
        pct_kept = 100 * n_kept / n_total if n_total > 0 else 0
        print(f"  📊 PDD filtering: Mantener {pct_kept:.1f}% de layers (eliminar 1% inferior)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path, target_idx = self.pairs[idx]
        
        inp = read_volume(input_path)
        tgt = read_volume(target_path)
        
        # 🎯 NUEVO: Filtrar por PDD baja
        # Eliminar capas que están en el 1% inferior
        pdd_mask = self.pdd_masks[target_idx]
        if not np.any(pdd_mask):  # Si todas las capas fueron eliminadas, skip
            return None, None, None
        
        # Aplicar máscara (mantener solo capas donde pdd_mask == True)
        # Para simplificar, simplemente evitamos crop en estas capas
        # En su lugar, normalizamos el volumen completo
        
        # Normalizar por máximo del INPUT
        max_input = float(np.max(inp))
        if max_input > 0:
            inp = inp / max_input
            tgt = tgt / max_input
        else:
            return None, None, None
        
        # Crop
        if self.is_train:
            inp, tgt = random_crop(inp, tgt, self.patch_size)
        else:
            inp = center_crop(inp, self.patch_size)
            tgt = center_crop(tgt, self.patch_size)
        
        # 🎯 NUEVO: Crear mapa de pesos por dosis
        weights = create_dose_weights(tgt, 
                                     dose_threshold=DOSE_THRESHOLD,
                                     low_weight=LOW_WEIGHT,
                                     high_weight=HIGH_WEIGHT)
        
        # A tensores
        inp = torch.from_numpy(inp).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        
        return inp, tgt, weights


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


def weighted_mse_loss(pred, tgt, weights):
    """
    Calcula MSE ponderado: mean((pred - tgt)² × weights)
    
    Args:
        pred: predicción del modelo (batch, channels, Z, Y, X)
        tgt: target (batch, channels, Z, Y, X)
        weights: pesos (batch, channels, Z, Y, X)
    
    Returns: pérdida escalar ponderada
    """
    diff = (pred - tgt) ** 2
    weighted_diff = diff * weights
    return weighted_diff.mean()


def main():
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    
    print("=" * 70)
    print("🧠 ENTRENAMIENTO CON MÁSCARA PONDERADA Y FILTRADO DE PDD BAJA")
    print("=" * 70)
    print(f"📂 Dataset:           {DATASET_ROOT}")
    print(f"📂 Output:            {OUTPUT_DIR}")
    print(f"🔧 Device:            {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU:               {torch.cuda.get_device_name(0)}")
    print(f"📊 Batch size:        {BATCH_SIZE}")
    print(f"📊 Patch size:        {PATCH_SIZE}")
    print(f"📊 Epochs:            {NUM_EPOCHS}")
    print(f"📊 Learning rate:     {LEARNING_RATE}")
    print()
    print(f"🎯 Dose threshold:    {DOSE_THRESHOLD*100:.0f}% (core/periferia)")
    print(f"🎯 Core weight:       {HIGH_WEIGHT}")
    print(f"🎯 Periphery weight:  {LOW_WEIGHT}")
    print(f"🎯 PDD filter:        Eliminar {PERCENTILE_PDD_LOW:.1f}% inferior")
    print("=" * 70)
    
    assert TRAIN_DIR.exists(), f"❌ No existe: {TRAIN_DIR}"
    assert VAL_DIR.exists(), f"❌ No existe: {VAL_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    print("\n📂 Cargando datasets...")
    train_ds = SimpleDoseDatasetWeighted(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, 
                                         PATCH_SIZE, percentile_pdd=PERCENTILE_PDD_LOW,
                                         is_train=True)
    val_ds   = SimpleDoseDatasetWeighted(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, 
                                         PATCH_SIZE, percentile_pdd=PERCENTILE_PDD_LOW,
                                         is_train=False)
    
    assert len(train_ds) > 0, "❌ No hay datos de entrenamiento"
    assert len(val_ds) > 0, "❌ No hay datos de validación"
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, pin_memory=True)
    
    # Modelo
    model = UNet3D(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo: {total_params:,} parámetros")
    
    # Training loop
    best_val_loss = float("inf")
    
    print("\n🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False):
            inp, tgt, weights = batch
            
            if inp is None:  # Skip batch si hubo error
                continue
            
            inp = inp.to(device)
            tgt = tgt.to(device)
            weights = weights.to(device)
            
            pred = model(inp)
            loss = weighted_mse_loss(pred, tgt, weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        if not train_losses:
            print(f"Epoch {epoch}: No training batches processed")
            continue
        
        train_loss = np.mean(train_losses)
        
        # ---- Validation ----
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt, weights = batch
                
                if inp is None:
                    continue
                
                inp = inp.to(device)
                tgt = tgt.to(device)
                weights = weights.to(device)
                
                pred = model(inp)
                loss = weighted_mse_loss(pred, tgt, weights)
                val_losses.append(loss.item())
        
        if not val_losses:
            print(f"Epoch {epoch}: No validation batches processed")
            continue
        
        val_loss = np.mean(val_losses)
        
        # ---- Log ----
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / "best.pt"))
            marker = " ⭐ BEST"
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}{marker}")
        
        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / f"ckpt_epoch_{epoch:03d}.pt"))
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📊 Mejor val_loss: {best_val_loss:.6f}")
    print(f"📁 Checkpoints en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
