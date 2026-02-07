#!/usr/bin/env python3
"""
train_v2.py - Entrenamiento con Residual Learning + Weighted Loss (Paper)
=========================================================================
Cambios fundamentales vs v1 (que predecía constante):

1. RESIDUAL LEARNING: modelo predice corrección, no dosis absoluta
   output = input + model(input)  →  modelo solo aprende el "delta"
   
2. EXPONENTIAL WEIGHTED LOSS (del artículo):
   W(Y) = exp[-α * (1 - 0.5*(Ŷ+Y) / max(Y))]
   Peso continuo que enfoca la red en zonas de alta dosis.
   Ignora voxels vacíos (96%). α=10 por default.
   
3. SMART PATCHES: preferir parches que contengan dosis real
   Evita entrenar con parches 100% vacíos
"""
# ---- Desactivar MIOpen (Yuca no lo soporta) ----
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
# -----------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from tqdm import tqdm

torch.backends.cudnn.enabled = False

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 2
NUM_EPOCHS = 100  # Más épocas: el residual learning converge más estable
LEARNING_RATE = 5e-4  # Más bajo para residual (los gradientes son más estables)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 16
PATCH_SIZE = 96

DATASET_ROOT = Path("dataset_pilot")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
RESULTS_DIR = Path("runs/denoising_v2_residual")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]

print(f"✓ Device: {DEVICE}")
print(f"✓ Strategy: RESIDUAL LEARNING + EXPONENTIAL W(Y) LOSS (α=10)")

# ============================================================================
# 3D U-NET con RESIDUAL (output = input + model(input))
# ============================================================================
class ResidualUNet3D(nn.Module):
    """U-Net que predice el RESIDUAL (corrección) en vez de la dosis absoluta."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        self.final = nn.Conv3d(base_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        residual = self.final(d1)
        # ⚡ RESIDUAL: output = input + corrección
        return x + residual

# ============================================================================
# WEIGHTED LOSS del artículo: W(Y) = exp[-α * (1 - 0.5*(Ŷ+Y)/max(Y))]
# ============================================================================
class ExponentialWeightedLoss(nn.Module):
    """
    Función de peso del artículo de dose denoising:
    
        W(Y) = exp[ -α * (1 - 0.5*(pred_abs + target_abs) / max(target_abs)) ]
    
    IMPORTANTE: Los pesos W(Y) se calculan con datos ABSOLUTOS (no normalizados)
    para mantener magnitudes físicamente correctas. El error se calcula con
    datos normalizados para estabilidad numérica.
    
    - Cuando avg(pred_abs,target_abs) ≈ max(target_abs):  W ≈ exp(0) = 1.0  (peso máximo)
    - Cuando avg(pred_abs,target_abs) ≈ 0:               W ≈ exp(-α) ≈ 0   (peso mínimo)
    
    Un solo hiperparámetro α controla la agresividad del enfoque.
    α grande → ignora más las zonas de baja dosis
    α pequeño → trata todo más uniformemente
    
    Paper recomienda α ∈ [5, 10]. Default: α=10.
    """
    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred_norm, target_norm, pred_abs, target_abs):
        # Máscara: solo donde target_abs > 5% del máximo
        # Esto evita que voxels de muy baja dosis colapsen los pesos W
        max_dose = target_abs.max().detach()
        mask = target_abs > 0.05 * max_dose
        n_active = mask.sum()
        
        if n_active == 0:
            return torch.tensor(0.0, device=pred_norm.device, requires_grad=True), {
                'total': 0, 'n_active': 0, 'w_mean': 0, 'w_max': 0, 'w_min': 0
            }
        
        # W(Y) = exp[-α * (1 - 0.5*(Ŷ_abs + Y_abs) / max(Y_abs))]
        # Los pesos se calculan con datos ABSOLUTOS para magnitudes correctas
        avg_dose_abs = 0.5 * (pred_abs + target_abs)
        ratio = avg_dose_abs / (max_dose + 1e-8)
        weights = torch.exp(-self.alpha * (1.0 - ratio))
        
        # MSE ponderado con datos NORMALIZADOS (para estabilidad)
        # pero pesos calculados en escala absoluta
        error = (pred_norm - target_norm) ** 2
        weighted_error = (error * weights)[mask]
        loss = weighted_error.mean()
        
        # Stats para monitoreo
        w_active = weights[mask].detach()
        return loss, {
            'total': loss.item(),
            'n_active': n_active.item(),
            'w_mean': w_active.mean().item(),
            'w_max': w_active.max().item(),
            'w_min': w_active.min().item()
        }

# ============================================================================
# DATASET con SMART PATCHES
# ============================================================================
class SmartPatchDataset(Dataset):
    """
    Dataset que prefiere parches con dosis real.
    Para cada sample, encuentra el bounding box de la dosis > 0
    y extrae parches centrados en esa región.
    """
    def __init__(self, split_dir, dataset_root, input_levels, split='train', patch_size=96):
        self.split_dir = Path(split_dir)
        self.dataset_root = Path(dataset_root)
        self.input_levels = input_levels
        self.split = split
        self.patch_size = patch_size
        self.samples = []

        n_targets = len(list(self.dataset_root.glob("target_*")))
        print(f"✓ Found {n_targets} targets in {self.dataset_root}")

        pair_dirs = sorted(self.split_dir.glob("pair_*"))
        print(f"✓ Found {len(pair_dirs)} pairs in split '{split}'")
        print(f"  Loading all data into memory...")

        for pair_dir in pair_dirs:
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = self.dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            if not target_mhd.exists():
                continue

            try:
                target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_mhd))).astype(np.float32)
            except Exception:
                continue

            # Encontrar bounding box de la dosis (donde target > 1% del max)
            max_dose = target_vol.max()
            if max_dose == 0:
                continue
            dose_mask = target_vol > 0.01 * max_dose
            if dose_mask.sum() == 0:
                continue
            
            # Bounding box de la región con dosis
            coords = np.argwhere(dose_mask)
            bb_min = coords.min(axis=0)  # [z_min, y_min, x_min]
            bb_max = coords.max(axis=0)  # [z_max, y_max, x_max]

            for level in self.input_levels:
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                if not input_mhd.exists():
                    continue

                try:
                    input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_mhd))).astype(np.float32)
                except Exception:
                    continue

                self.samples.append({
                    'input': torch.from_numpy(input_vol).float(),
                    'target': torch.from_numpy(target_vol).float(),
                    'max_dose': max_dose,
                    'bb_min': bb_min,
                    'bb_max': bb_max,
                    'level': level,
                    'pair': pair_dir.name
                })

        print(f"✓ Loaded {len(self.samples)} samples into memory")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z, y, x = sample['target'].shape
        ps = self.patch_size
        bb_min = sample['bb_min']
        bb_max = sample['bb_max']

        # 80% de las veces: patch centrado en la región de dosis
        # 20% de las veces: patch aleatorio (para que aprenda a no añadir ruido fuera)
        if np.random.rand() < 0.8:
            # Centro aleatorio DENTRO del bounding box de dosis
            z_center = np.random.randint(bb_min[0], bb_max[0] + 1)
            y_center = np.random.randint(bb_min[1], bb_max[1] + 1)
            x_center = np.random.randint(bb_min[2], bb_max[2] + 1)
            
            z_start = max(0, min(z_center - ps // 2, z - ps))
            y_start = max(0, min(y_center - ps // 2, y - ps))
            x_start = max(0, min(x_center - ps // 2, x - ps))
        else:
            # Patch completamente aleatorio
            z_start = np.random.randint(0, max(z - ps, 0) + 1) if z > ps else 0
            y_start = np.random.randint(0, max(y - ps, 0) + 1) if y > ps else 0
            x_start = np.random.randint(0, max(x - ps, 0) + 1) if x > ps else 0

        z_end = min(z_start + ps, z)
        y_end = min(y_start + ps, y)
        x_end = min(x_start + ps, x)

        input_patch = sample['input'][z_start:z_end, y_start:y_end, x_start:x_end]
        target_patch = sample['target'][z_start:z_end, y_start:y_end, x_start:x_end]

        return {
            'input': input_patch.unsqueeze(0),
            'target': target_patch.unsqueeze(0),
            'max_dose': torch.tensor(sample['max_dose'], dtype=torch.float32)
        }

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    n_active_total = 0
    w_sum = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]", leave=True)
    for batch in pbar:
        input_data = batch['input'].to(device, non_blocking=True)
        target_data = batch['target'].to(device, non_blocking=True)
        max_dose = batch['max_dose'].to(device, non_blocking=True)

        # Normalizar a [0, 1]
        max_dose_view = max_dose.view(-1, 1, 1, 1, 1) + 1e-8
        input_norm = input_data / max_dose_view
        target_norm = target_data / max_dose_view

        optimizer.zero_grad()
        # ⚡ RESIDUAL: model(input) = input + corrección
        pred_norm = model(input_norm)
        # Desdenormalizar predicción para calcular pesos (pred_abs = pred_norm * max_dose)
        pred_abs = pred_norm * max_dose_view
        target_abs = target_data  # Ya están en escala absoluta
        loss, loss_stats = loss_fn(pred_norm, target_norm, pred_abs, target_abs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_active_total += loss_stats['n_active']
        w_sum += loss_stats['w_mean']
        n_batches += 1
        pbar.set_postfix(loss=f'{loss.item():.6f}', w=f'{loss_stats["w_mean"]:.3f}')

    avg_loss = total_loss / max(n_batches, 1)
    avg_w = w_sum / max(n_batches, 1)
    print(f"✓ Epoch {epoch+1} - Loss: {avg_loss:.6f} | W(avg): {avg_w:.4f}")
    print(f"  Voxels activos totales: {n_active_total:,}")
    return avg_loss, {'n_active': n_active_total, 'w_mean': avg_w}

def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [val]", leave=True)
    with torch.no_grad():
        for batch in pbar:
            input_data = batch['input'].to(device, non_blocking=True)
            target_data = batch['target'].to(device, non_blocking=True)
            max_dose = batch['max_dose'].to(device, non_blocking=True)

            max_dose_view = max_dose.view(-1, 1, 1, 1, 1) + 1e-8
            input_norm = input_data / max_dose_view
            target_norm = target_data / max_dose_view

            pred_norm = model(input_norm)
            pred_abs = pred_norm * max_dose_view
            target_abs = target_data
            loss, _ = loss_fn(pred_norm, target_norm, pred_abs, target_abs)
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.6f}')

    avg_loss = total_loss / len(dataloader)
    print(f"✓ Val Loss: {avg_loss:.6f}")
    return avg_loss

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("TRAIN V2: Residual U-Net + Masked Loss + Smart Patches")
    print("=" * 70)

    train_dataset = SmartPatchDataset(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, split='train', patch_size=PATCH_SIZE)
    val_dataset = SmartPatchDataset(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, split='val', patch_size=PATCH_SIZE)

    # Oversampling x10 para variabilidad de parches
    sampler = WeightedRandomSampler(
        torch.ones(len(train_dataset)),
        num_samples=len(train_dataset) * 10,
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  Train samples: {len(train_dataset)} × 10 oversampling = {len(train_dataset)*10} per epoch")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Iters/epoch: {len(train_loader)}")

    # Modelo RESIDUAL
    model = ResidualUNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: ResidualUNet3D ({params:,} params)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_fn = ExponentialWeightedLoss(alpha=10.0)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20  # Early stopping
    no_improve = 0

    print(f"\n  Starting {NUM_EPOCHS} epochs (early stopping patience={patience})...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f})")
        print(f"{'='*70}")

        train_loss, _ = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch)
        val_loss = validate(model, val_loader, loss_fn, DEVICE, epoch)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, RESULTS_DIR / 'best_model.pt')
            print(f"  ✓ Best model saved ({val_loss:.6f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\n  ⚠ Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), RESULTS_DIR / f'checkpoint_epoch_{epoch+1}.pt')

    # Save final
    torch.save(model.state_dict(), RESULTS_DIR / 'final_model.pt')
    with open(RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING V2 COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
    print(f"Results: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
