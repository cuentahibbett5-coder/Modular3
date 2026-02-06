#!/usr/bin/env python3
"""
=============================================================
ENTRENAMIENTO CON SSIM LOSS (no MSE)
=============================================================
Usa parches 64³ pero SSIM Loss en lugar de MSE.
SSIM prioriza estructura, no valores absolutos.
"""

import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk

torch.backends.cudnn.enabled = False

# =============================================================
# SSIM Loss
# =============================================================
class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss (1 - SSIM)"""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.create_window()
    
    def create_window(self):
        gauss = torch.Tensor([np.exp(-(x - self.window_size//2)**2 / (2*self.sigma**2)) 
                             for x in range(self.window_size)])
        gauss = gauss / gauss.sum()
        self.register_buffer('window', gauss.unsqueeze(1).unsqueeze(1).unsqueeze(1))
    
    def forward(self, x, y):
        """SSIM between x and y (both shape [batch, 1, d, h, w], values in [0,1])"""
        # Constantes
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convolucionar con ventana gaussiana (x ya es [batch, 1, d, h, w])
        mu_x = F.conv3d(x, self.window.to(x.device), padding=self.window_size//2)
        mu_y = F.conv3d(y, self.window.to(x.device), padding=self.window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv3d(x ** 2, self.window.to(x.device), padding=self.window_size//2) - mu_x_sq
        sigma_y_sq = F.conv3d(y ** 2, self.window.to(x.device), padding=self.window_size//2) - mu_y_sq
        sigma_xy = F.conv3d(x * y, self.window.to(x.device), padding=self.window_size//2) - mu_xy
        
        # SSIM
        ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1)*(sigma_x_sq + sigma_y_sq + C2))
        return 1 - ssim_map.mean()


# =============================================================
# ⚙️ VARIABLES DE CONFIGURACIÓN
# =============================================================

DATASET_ROOT   = Path("dataset_pilot")
TRAIN_DIR      = DATASET_ROOT / "train"
VAL_DIR        = DATASET_ROOT / "val"
OUTPUT_DIR     = Path("runs/denoising_ssim")

INPUT_LEVELS   = ["input_1M", "input_2M", "input_5M", "input_10M"]

BATCH_SIZE     = 2
PATCH_SIZE     = (64, 64, 64)
NUM_EPOCHS     = 100
LEARNING_RATE  = 1e-3
DEVICE         = "auto"
USE_AMP        = True

# =============================================================

def read_volume(mhd_path: Path) -> np.ndarray:
    npy_path = mhd_path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    return arr.astype(np.float32)


def random_crop(vol_a, vol_b, patch_size):
    z, y, x = vol_a.shape
    pz, py, px = patch_size
    sz = np.random.randint(0, max(z - pz + 1, 1))
    sy = np.random.randint(0, max(y - py + 1, 1))
    sx = np.random.randint(0, max(x - px + 1, 1))
    return (vol_a[sz:sz+pz, sy:sy+py, sx:sx+px],
            vol_b[sz:sz+pz, sy:sy+py, sx:sx+px])


def center_crop(vol, patch_size):
    z, y, x = vol.shape
    pz, py, px = patch_size
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return vol[sz:sz+pz, sy:sy+py, sx:sx+px]


class SimpleDoseDataset(Dataset):
    def __init__(self, split_dir, dataset_root, levels, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train
        self.pairs = []
        
        n_targets = len(list(dataset_root.glob("target_*")))
        
        for pair_dir in sorted(split_dir.glob("pair_*")):
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            
            if not target_mhd.exists():
                continue
            
            for level in levels:
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                
                if input_mhd.exists():
                    self.pairs.append((input_mhd, target_mhd))
        
        print(f"  ✅ {len(self.pairs)} pares encontrados en {split_dir.name}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        
        inp = read_volume(input_path)
        tgt = read_volume(target_path)
        
        max_val = float(np.max(tgt))
        if max_val > 0:
            inp = inp / max_val
            tgt = tgt / max_val
        
        if self.is_train:
            inp, tgt = random_crop(inp, tgt, self.patch_size)
        else:
            inp = center_crop(inp, self.patch_size)
            tgt = center_crop(tgt, self.patch_size)
        
        inp = torch.from_numpy(inp).unsqueeze(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0)
        
        return inp, tgt


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


def main():
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    
    print("=" * 60)
    print("🧠 ENTRENAMIENTO CON SSIM LOSS")
    print("=" * 60)
    print(f"📂 Dataset:     {DATASET_ROOT}")
    print(f"🔧 Device:      {device}")
    if torch.cuda.is_available():
        print(f"🔧 GPU:         {torch.cuda.get_device_name(0)}")
    print(f"📊 Batch size:  {BATCH_SIZE}")
    print(f"📊 Patch size:  {PATCH_SIZE}")
    print(f"📊 Epochs:      {NUM_EPOCHS}")
    print(f"📊 Loss:        SSIM (prioriza estructura)")
    print("=" * 60)
    
    assert TRAIN_DIR.exists(), f"❌ No existe: {TRAIN_DIR}"
    assert VAL_DIR.exists(), f"❌ No existe: {VAL_DIR}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Cargando datasets...")
    train_ds = SimpleDoseDataset(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, PATCH_SIZE, is_train=True)
    val_ds   = SimpleDoseDataset(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, PATCH_SIZE, is_train=False)
    
    assert len(train_ds) > 0, "❌ No hay datos de entrenamiento"
    assert len(val_ds) > 0, "❌ No hay datos de validación"
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    model = UNet3D(base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = SSIMLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Modelo: {total_params:,} parámetros")
    
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    best_val_loss = float("inf")
    
    print("\n🚀 Iniciando entrenamiento...\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    pred = model(inp)
                    loss = loss_fn(pred, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)
                
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        pred = model(inp)
                        loss = loss_fn(pred, tgt)
                else:
                    pred = model(inp)
                    loss = loss_fn(pred, tgt)
                
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        # Log
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
