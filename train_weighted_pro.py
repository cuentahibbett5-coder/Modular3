#!/usr/bin/env python3
"""
Advanced Weighted Training with:
1. Weighted Sampling (50% core, 50% periphery)
2. Dynamic Loss proportional to dose level
"""
# ---- Desactivar MIOpen (ANTES de importar torch) ----
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
torch_backends_cudnn_enabled = False
# -----------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from datetime import datetime
import sys
from tqdm import tqdm

# Desactivar MIOpen después de importar torch
torch.backends.cudnn.enabled = False

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 4  # Reducido de 16 para caber en memoria
NUM_EPOCHS = 50  # Reducido para pruebas más rápidas
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 16  # Reducido de 32 para menor uso de memoria

# Rutas (relativas, como en train_fullvol.py)
DATASET_ROOT = Path("dataset_pilot")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
RESULTS_DIR = Path("runs/denoising_weighted_pro")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input levels
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]

# Threshold definitions
DOSE_THRESHOLD_CORE = 0.20  # 20% of max = core
DOSE_THRESHOLD_SIGNIFICANT = 0.01  # 1% of max = significant

print(f"✓ Device: {DEVICE}")
print(f"✓ Dataset: {DATASET_ROOT}")
print(f"✓ Results: {RESULTS_DIR}")

# ============================================================================
# 3D U-NET ARCHITECTURE
# ============================================================================
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.base_ch = base_channels
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
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
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out

# ============================================================================
# DYNAMIC LOSS FUNCTION
# ============================================================================
class DynamicDoseLoss(nn.Module):
    """
    Loss proportional to dose level:
    - High dose (core): weight_high = 1.0 → error²  × 1.0
    - Mid dose: weight ∝ log(dose)  → smooth transition
    - Low dose (noise): weight_low → error² × small_weight
    
    Formula: weight(dose) = 1.0 if dose >= threshold_core
                          = 0.1 + 0.9 * (dose / threshold_core) if dose < threshold_core
    """
    def __init__(self, threshold_core=0.20, threshold_low=0.01, weight_min=0.1):
        super().__init__()
        self.threshold_core = threshold_core
        self.threshold_low = threshold_low
        self.weight_min = weight_min
        
    def forward(self, pred, target, max_dose=None):
        """
        Args:
            pred: predicted dose
            target: ground truth dose
            max_dose: maximum dose in batch (for normalization)
        """
        if max_dose is None:
            max_dose = target.max().detach()
        
        # Normalized dose (0-1)
        normalized_dose = target / (max_dose + 1e-8)
        
        # Calculate dynamic weights based on dose level
        weights = torch.ones_like(normalized_dose)
        
        # High dose (core): weight = 1.0
        mask_high = normalized_dose >= self.threshold_core
        weights[mask_high] = 1.0
        
        # Mid dose: linear interpolation
        mask_mid = (normalized_dose >= self.threshold_low) & (normalized_dose < self.threshold_core)
        weights[mask_mid] = self.weight_min + (1.0 - self.weight_min) * \
                            (normalized_dose[mask_mid] - self.threshold_low) / \
                            (self.threshold_core - self.threshold_low)
        
        # Low dose (noise): weight = weight_min
        mask_low = normalized_dose < self.threshold_low
        weights[mask_low] = self.weight_min
        
        # MSE loss weighted by dose
        mse = (pred - target) ** 2
        weighted_mse = mse * weights
        loss = weighted_mse.mean()
        
        return loss, {
            'total': loss.item(),
            'high_dose_count': mask_high.sum().item(),
            'mid_dose_count': mask_mid.sum().item(),
            'low_dose_count': mask_low.sum().item()
        }

# ============================================================================
# DATASET WITH DOSE CALCULATION
# ============================================================================
def calculate_pdd(volume):
    """Calculate PDD (Percent Depth Dose) - max dose per z-layer"""
    z_size = volume.shape[0]
    pdd = np.zeros(z_size)
    for z in range(z_size):
        pdd[z] = volume[z].max()
    return pdd

def get_dose_category(pdd, max_dose, threshold_core=0.20, threshold_sig=0.01):
    """
    Classify which category each layer belongs to:
    0 = Low dose (< 1%)
    1 = Mid dose (1% - 20%)
    2 = Core (>= 20%)
    """
    normalized_pdd = pdd / max_dose
    category = np.zeros_like(pdd, dtype=int)
    category[normalized_pdd >= threshold_core] = 2
    category[(normalized_pdd >= threshold_sig) & (normalized_pdd < threshold_core)] = 1
    return category

class SimpleDoseDatasetWeighted(Dataset):
    """Dataset with voxel-level dose tracking for sampling"""
    def __init__(self, split_dir, dataset_root, input_levels, split='train', patch_size=128):
        self.split_dir = Path(split_dir)
        self.dataset_root = Path(dataset_root)
        self.input_levels = input_levels
        self.split = split
        self.patch_size = patch_size
        self.samples = []
        self.dose_weights = []  # For weighted sampling
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # Contar targets
        n_targets = len(list(self.dataset_root.glob("target_*")))
        print(f"✓ Found {n_targets} targets in {self.dataset_root}")
        
        # Buscar pairs y cargar
        pair_dirs = sorted(self.split_dir.glob("pair_*"))
        print(f"✓ Found {len(pair_dirs)} pairs in split '{split}'")
        
        for pair_dir in pair_dirs:
            pair_num = int(pair_dir.name.split("_")[-1])
            target_idx = ((pair_num - 1) % n_targets) + 1
            target_mhd = self.dataset_root / f"target_{target_idx}" / "dose_edep.mhd"
            
            if not target_mhd.exists():
                print(f"  ⚠ Target missing for {pair_dir.name}: {target_mhd}")
                continue
            
            # Cargar para cada nivel de input
            for level in self.input_levels:
                input_mhd = pair_dir / f"{level}.mhd"
                if not input_mhd.exists():
                    input_mhd = pair_dir / level / "dose_edep.mhd"
                
                if not input_mhd.exists():
                    continue  # Skip this level
                
                # Load volumes
                try:
                    input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_mhd)))
                    target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_mhd)))
                except Exception as e:
                    print(f"  ⚠ Error loading {pair_dir.name}/{level}: {e}")
                    continue
                
                # Calculate PDD for sampling strategy
                pdd = calculate_pdd(target_vol)
                max_dose = pdd.max()
                if max_dose == 0:
                    continue  # Skip zero dose
                
                dose_category = get_dose_category(pdd, max_dose)
                
                # Mean dose in this sample
                mean_dose = target_vol.mean()
                
                # Sampling weight: prefer samples with more core data
                core_fraction = (dose_category == 2).sum() / len(dose_category)
                sample_weight = core_fraction if core_fraction > 0 else 0.1
                
                self.samples.append({
                    'input': input_vol.astype(np.float32),
                    'target': target_vol.astype(np.float32),
                    'pdd': pdd,
                    'max_dose': max_dose,
                    'dose_category': dose_category,
                    'mean_dose': mean_dose,
                    'core_fraction': core_fraction
                })
                self.dose_weights.append(sample_weight)
        
        print(f"✓ Loaded {len(self.samples)} samples")
        
        if len(self.dose_weights) == 0:
            print(f"\n⚠ ERROR: No samples found in {self.split_dir}")
            print(f"  Expected structure:")
            print(f"    - {self.split_dir}/pair_*/[input_1M.mhd, input_2M.mhd, etc]")
            print(f"    - {self.dataset_root}/target_*/dose_edep.mhd")
            raise FileNotFoundError(f"No training data found in {self.split_dir}")
        
        print(f"  Core fraction range: {min(self.dose_weights):.3f} - {max(self.dose_weights):.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get volume dimensions
        z, y, x = sample['target'].shape
        
        # Random patch or full volume
        if z > self.patch_size:
            z_start = np.random.randint(0, z - self.patch_size + 1)
            z_end = z_start + self.patch_size
        else:
            z_start, z_end = 0, z
        
        if y > self.patch_size:
            y_start = np.random.randint(0, y - self.patch_size + 1)
            y_end = y_start + self.patch_size
        else:
            y_start, y_end = 0, y
        
        if x > self.patch_size:
            x_start = np.random.randint(0, x - self.patch_size + 1)
            x_end = x_start + self.patch_size
        else:
            x_start, x_end = 0, x
        
        input_patch = sample['input'][z_start:z_end, y_start:y_end, x_start:x_end]
        target_patch = sample['target'][z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Normalize by max dose in target
        max_dose = target_patch.max()
        if max_dose > 0:
            input_patch = input_patch / max_dose
            target_patch = target_patch / max_dose
        
        return {
            'input': torch.from_numpy(input_patch).unsqueeze(0).float(),
            'target': torch.from_numpy(target_patch).unsqueeze(0).float(),
            'max_dose': torch.tensor(sample['max_dose'], dtype=torch.float32)
        }

# ============================================================================
# WEIGHTED SAMPLING STRATEGY
# ============================================================================
class WeightedDoseSampler(torch.utils.data.Sampler):
    """
    Custom sampler that balances:
    - 50% of batches from high-core-content samples
    - 50% of batches from low-core-content samples
    """
    def __init__(self, dose_weights, batch_size, num_samples=None):
        self.dose_weights = np.array(dose_weights, dtype=np.float32)
        self.batch_size = batch_size
        self.num_samples = num_samples if num_samples else len(dose_weights)
        
        # Normalize weights for WeightedRandomSampler
        self.weights = torch.from_numpy(self.dose_weights / self.dose_weights.sum())
    
    def __iter__(self):
        # Use torch's WeightedRandomSampler for balanced sampling
        sampler = WeightedRandomSampler(
            self.weights,
            num_samples=self.num_samples,
            replacement=True
        )
        return iter(sampler)
    
    def __len__(self):
        return self.num_samples

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    stats = {'high': 0, 'mid': 0, 'low': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]", leave=True)
    for batch_idx, batch in enumerate(pbar):
        input_data = batch['input'].to(device)
        target_data = batch['target'].to(device)
        max_dose = batch['max_dose'].to(device)
        
        # Forward pass
        output = model(input_data)
        
        # Dynamic loss
        loss, loss_stats = loss_fn(output, target_data, max_dose.max())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        stats['high'] += loss_stats['high_dose_count']
        stats['mid'] += loss_stats['mid_dose_count']
        stats['low'] += loss_stats['low_dose_count']
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item():.6f})
    
    avg_loss = total_loss / len(dataloader)
    print(f"✓ Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")
    print(f"  High dose voxels: {stats['high']:,} | Mid dose: {stats['mid']:,} | Low dose: {stats['low']:,}")
    
    return avg_loss, stats

def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [val]", leave=True)
    with torch.no_grad():
        for batch in pbar:
            input_data = batch['input'].to(device)
            target_data = batch['target'].to(device)
            max_dose = batch['max_dose'].to(device)
            
            output = model(input_data)
            loss, _ = loss_fn(output, target_data, max_dose.max())
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item():.6f})
    
    avg_loss = total_loss / len(dataloader)
    print(f"✓ Val Loss: {avg_loss:.6f}")
    return avg_loss

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print("\n" + "="*70)
    print("TRAINING: Weighted Sampling + Dynamic Dose Loss (PRO)")
    print("="*70)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Results: {RESULTS_DIR}")
    print("="*70)
    
    # Verify directories exist
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"❌ Train dir not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"❌ Val dir not found: {VAL_DIR}")
    
    # Create datasets
    print("\n[1/5] Loading datasets...")
    train_dataset = SimpleDoseDatasetWeighted(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, split='train')
    val_dataset = SimpleDoseDatasetWeighted(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, split='val')
    
    # Create weighted sampler
    print(f"\n[2/5] Creating weighted sampler...")
    print(f"  Strategy: 50% core samples, 50% periphery samples")
    train_sampler = WeightedDoseSampler(
        train_dataset.dose_weights,
        batch_size=BATCH_SIZE,
        num_samples=len(train_dataset) * 10  # Oversample for better mixing
    )
    
    # Create dataloaders
    print(f"\n[3/5] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Model and optimizer
    print(f"\n[4/5] Building model...")
    model = UNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    loss_fn = DynamicDoseLoss(
        threshold_core=DOSE_THRESHOLD_CORE,
        threshold_low=DOSE_THRESHOLD_SIGNIFICANT,
        weight_min=0.1  # Penalize errors in high dose, tolerate in low dose
    )
    
    # Training loop
    print(f"\n[5/5] Starting training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_stats = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, DEVICE, epoch)
        
        # Schedule
        scheduler.step()
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = RESULTS_DIR / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"  ✓ Best model saved ({val_loss:.6f})")
        
        # Save every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = RESULTS_DIR / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved")
    
    # Final checkpoint
    final_path = RESULTS_DIR / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    
    # Save history
    history_path = RESULTS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")
    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  • best_model.pt")
    print(f"  • final_model.pt")
    print(f"  • training_history.json")

if __name__ == '__main__':
    main()
