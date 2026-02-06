#!/usr/bin/env python3
"""
=============================================================
SCRIPT DE ENTRENAMIENTO SIMPLE - Denoising Monte Carlo
=============================================================
Revisa las variables al inicio antes de ejecutar.
"""

# ---- Fix MIOpen/ROCm para AMD GPUs (ANTES de importar torch) ----
import os
_tmp = f"/tmp/miopen_{os.environ.get('USER', 'user')}"
os.makedirs(_tmp, exist_ok=True)
os.environ["MIOPEN_USER_DB_PATH"]        = _tmp
os.environ["MIOPEN_CACHE_DIR"]           = _tmp
os.environ["TMPDIR"]                     = "/tmp"
os.environ["MIOPEN_FIND_ENFORCE"]         = "3"      # IMMEDIATE: usa algoritmo por defecto, sin búsqueda
os.environ["HSA_FORCE_FINE_GRAIN_PCIE"]   = "1"
# -----------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

# =============================================================
# ⚙️ VARIABLES DE CONFIGURACIÓN - REVISAR ANTES DE EJECUTAR
# =============================================================

# Rutas
DATASET_ROOT   = Path("dataset_pilot")          # Carpeta raíz del dataset
TRAIN_DIR      = DATASET_ROOT / "train"          # Subcarpeta de entrenamiento
VAL_DIR        = DATASET_ROOT / "val"            # Subcarpeta de validación
OUTPUT_DIR     = Path("runs/denoising_v2")       # Donde se guardan los checkpoints

# Niveles de input (carpetas o archivos dentro de cada pair)
INPUT_LEVELS   = ["input_1M", "input_2M", "input_5M", "input_10M"]

# Hiperparámetros
BATCH_SIZE     = 2                # Tamaño de batch (reducido para evitar errores GPU)
PATCH_SIZE     = (64, 64, 64)     # Tamaño de los patches 3D
NUM_EPOCHS     = 50               # Número de épocas
LEARNING_RATE  = 1e-3             # Learning rate
DEVICE         = "auto"           # "auto", "cuda" o "cpu"

# =============================================================
# FIN DE CONFIGURACIÓN
# =============================================================


def read_volume(mhd_path: Path) -> np.ndarray:
    """Lee un archivo .mhd y retorna un array 3D float32."""
    # Intentar .npy con el mismo nombre base
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    
    # Leer con SimpleITK
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


def random_crop(vol_a, vol_b, patch_size):
    """Crop aleatorio sincronizado de dos volúmenes."""
    z, y, x = vol_a.shape
    pz, py, px = patch_size
    sz = np.random.randint(0, max(z - pz + 1, 1))
    sy = np.random.randint(0, max(y - py + 1, 1))
    sx = np.random.randint(0, max(x - px + 1, 1))
    return (vol_a[sz:sz+pz, sy:sy+py, sx:sx+px],
            vol_b[sz:sz+pz, sy:sy+py, sx:sx+px])


def center_crop(vol, patch_size):
    """Crop central de un volumen."""
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


class SimpleDoseDataset(Dataset):
    """Dataset simple de pares input/target."""
    
    def __init__(self, split_dir, dataset_root, levels, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.pairs = []  # Lista de (input_path, target_path)
        
        n_targets = len(list(dataset_root.glob("target_*")))
        
        for pair_dir in sorted(split_dir.glob("pair_*")):
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            
            if not target_mhd.exists():
                print(f"  ⚠️  Target no encontrado: {target_mhd}")
                continue
            
            for level in levels:
                # Buscar input: primero como archivo, luego como subdirectorio
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                
                if input_mhd.exists():
                    self.pairs.append((input_mhd, target_mhd))
                else:
                    print(f"  ⚠️  Input no encontrado: {pair_dir.name}/{level}")
        
        print(f"  ✅ {len(self.pairs)} pares encontrados en {split_dir.name}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        
        inp = read_volume(input_path)
        tgt = read_volume(target_path)
        
        # Normalizar por el máximo del target
        max_val = float(np.max(tgt))
        if max_val > 0:
            inp = inp / max_val
            tgt = tgt / max_val
        
        # Crop
        if self.is_train:
            inp, tgt = random_crop(inp, tgt, self.patch_size)
        else:
            inp = center_crop(inp, self.patch_size)
            tgt = center_crop(tgt, self.patch_size)
        
        # A tensores con canal
        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, Z, Y, X)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        
        return inp, tgt


# ---- Modelo UNet3D (igual que antes) ----

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


# ---- Entrenamiento ----

def main():
    # Device
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    
    print("=" * 60)
    print("🧠 ENTRENAMIENTO DE DENOISING")
    print("=" * 60)
    print(f"📂 Dataset:     {DATASET_ROOT}")
    print(f"📂 Train dir:   {TRAIN_DIR}")
    print(f"📂 Val dir:     {VAL_DIR}")
    print(f"📂 Output:      {OUTPUT_DIR}")
    print(f"🔧 Device:      {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU:         {torch.cuda.get_device_name(0)}")
    print(f"📊 Batch size:  {BATCH_SIZE}")
    print(f"📊 Patch size:  {PATCH_SIZE}")
    print(f"📊 Epochs:      {NUM_EPOCHS}")
    print(f"📊 LR:          {LEARNING_RATE}")
    print("=" * 60)
    
    # Verificar que existen las carpetas
    assert TRAIN_DIR.exists(), f"❌ No existe: {TRAIN_DIR}"
    assert VAL_DIR.exists(), f"❌ No existe: {VAL_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    print("\n📂 Cargando datasets...")
    train_ds = SimpleDoseDataset(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, PATCH_SIZE, is_train=True)
    val_ds   = SimpleDoseDataset(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, PATCH_SIZE, is_train=False)
    
    assert len(train_ds) > 0, "❌ No hay datos de entrenamiento"
    assert len(val_ds) > 0, "❌ No hay datos de validación"
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Modelo
    model = UNet3D(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo: {total_params:,} parámetros")
    
    # ---- Warmup de GPU (MIOpen necesita compilar kernels la primera vez) ----
    if device.type == "cuda":
        print("\n🔍 Warmup de GPU (puede tardar ~30s la primera vez)...")
        # Warmup progresivo: conv3d pequeño → modelo completo
        _w = torch.randn(1, 1, 16, 16, 16, device=device)
        _c = nn.Conv3d(1, 8, 3, padding=1).to(device)
        with torch.no_grad():
            _ = _c(_w)
        del _w, _c
        # Ahora probar modelo completo
        _test = torch.randn(1, 1, 64, 64, 64, device=device)
        with torch.no_grad():
            _ = model(_test)
        del _test
        torch.cuda.empty_cache()
        print("   ✅ GPU lista")
    
    # Training loop
    best_val_loss = float("inf")
    
    print("\n🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # ---- Validation ----
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        # ---- Log ----
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / "best.pt"))
            marker = " ⭐ BEST"
        
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}{marker}")
        
        # Guardar checkpoint cada 10 épocas
        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       str(OUTPUT_DIR / f"ckpt_epoch_{epoch:03d}.pt"))
    
    print(f"\n✅ Entrenamiento completado!")
    print(f"📊 Mejor val_loss: {best_val_loss:.6f}")
    print(f"📁 Checkpoints en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
