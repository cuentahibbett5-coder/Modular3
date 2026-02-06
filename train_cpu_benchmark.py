#!/usr/bin/env python3
"""
CPU-ONLY BENCHMARK: Misma arquitectura que train_weighted_pro.py
pero forzando CPU para comparar velocidad vs GPU (cudnn off).
Solo 5 épocas para medir s/it.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Forzar CPU, sin GPU

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import time
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 4
NUM_EPOCHS = 5  # Solo 5 épocas para benchmark
LEARNING_RATE = 1e-3
DEVICE = torch.device('cpu')  # FORZAR CPU
BASE_CHANNELS = 16

DATASET_ROOT = Path("dataset_pilot")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
RESULTS_DIR = Path("runs/cpu_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
DOSE_THRESHOLD_CORE = 0.20
DOSE_THRESHOLD_SIGNIFICANT = 0.01

print(f"{'='*70}")
print(f"CPU BENCHMARK - Comparación de velocidad")
print(f"{'='*70}")
print(f"✓ Device: {DEVICE} (CPU forzado)")
print(f"✓ Epochs: {NUM_EPOCHS} (solo benchmark)")
print(f"✓ Batch size: {BATCH_SIZE}")

# ============================================================================
# 3D U-NET (idéntica)
# ============================================================================
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.base_ch = base_channels
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
        return self.final(d1)

# ============================================================================
# DYNAMIC LOSS (idéntica)
# ============================================================================
class DynamicDoseLoss(nn.Module):
    def __init__(self, threshold_core=0.20, threshold_low=0.01, weight_min=0.1):
        super().__init__()
        self.threshold_core = threshold_core
        self.threshold_low = threshold_low
        self.weight_min = weight_min

    def forward(self, pred, target, max_dose=None):
        if max_dose is None:
            max_dose = target.max().detach()
        normalized_dose = target / (max_dose + 1e-8)
        weights = torch.ones_like(normalized_dose)
        mask_high = normalized_dose >= self.threshold_core
        weights[mask_high] = 1.0
        mask_mid = (normalized_dose >= self.threshold_low) & (normalized_dose < self.threshold_core)
        weights[mask_mid] = self.weight_min + (1.0 - self.weight_min) * \
                            (normalized_dose[mask_mid] - self.threshold_low) / \
                            (self.threshold_core - self.threshold_low)
        mask_low = normalized_dose < self.threshold_low
        weights[mask_low] = self.weight_min
        mse = (pred - target) ** 2
        loss = (mse * weights).mean()
        return loss, {
            'total': loss.item(),
            'high_dose_count': mask_high.sum().item(),
            'mid_dose_count': mask_mid.sum().item(),
            'low_dose_count': mask_low.sum().item()
        }

# ============================================================================
# DATASET (idéntica)
# ============================================================================
def calculate_pdd(volume):
    z_size = volume.shape[0]
    pdd = np.zeros(z_size)
    for z in range(z_size):
        pdd[z] = volume[z].max()
    return pdd

def get_dose_category(pdd, max_dose, threshold_core=0.20, threshold_sig=0.01):
    normalized_pdd = pdd / max_dose
    category = np.zeros_like(pdd, dtype=int)
    category[normalized_pdd >= threshold_core] = 2
    category[(normalized_pdd >= threshold_sig) & (normalized_pdd < threshold_core)] = 1
    return category

class SimpleDoseDatasetWeighted(Dataset):
    def __init__(self, split_dir, dataset_root, input_levels, split='train', patch_size=128):
        self.split_dir = Path(split_dir)
        self.dataset_root = Path(dataset_root)
        self.input_levels = input_levels
        self.patch_size = patch_size
        self.samples = []
        self.dose_weights = []

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

                pdd = calculate_pdd(target_vol)
                max_dose = pdd.max()
                if max_dose == 0:
                    continue
                dose_category = get_dose_category(pdd, max_dose)
                core_fraction = (dose_category == 2).sum() / len(dose_category)

                self.samples.append({
                    'input': torch.from_numpy(input_vol).float(),
                    'target': torch.from_numpy(target_vol).float(),
                    'max_dose': max_dose,
                    'core_fraction': core_fraction
                })
                self.dose_weights.append(core_fraction if core_fraction > 0 else 0.1)

        print(f"✓ Loaded {len(self.samples)} samples into memory")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        z, y, x = sample['target'].shape
        ps = self.patch_size

        z_s = np.random.randint(0, max(z - ps, 0) + 1) if z > ps else 0
        y_s = np.random.randint(0, max(y - ps, 0) + 1) if y > ps else 0
        x_s = np.random.randint(0, max(x - ps, 0) + 1) if x > ps else 0
        z_e, y_e, x_e = min(z_s + ps, z), min(y_s + ps, y), min(x_s + ps, x)

        return {
            'input': sample['input'][z_s:z_e, y_s:y_e, x_s:x_e].unsqueeze(0),
            'target': sample['target'][z_s:z_e, y_s:y_e, x_s:x_e].unsqueeze(0),
            'max_dose': torch.tensor(sample['max_dose'], dtype=torch.float32)
        }

# ============================================================================
# TRAINING (CPU, sin AMP)
# ============================================================================
def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [CPU train]", leave=True)
    for batch in pbar:
        input_data = batch['input'].to(device)
        target_data = batch['target'].to(device)
        max_dose = batch['max_dose'].to(device)

        max_dose_view = max_dose.view(-1, 1, 1, 1, 1) + 1e-8
        input_data = input_data / max_dose_view
        target_data = target_data / max_dose_view

        optimizer.zero_grad()
        output = model(input_data)
        loss, _ = loss_fn(output, target_data, max_dose.max())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.6f}')

    avg_loss = total_loss / len(dataloader)
    print(f"✓ Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")
    return avg_loss

def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [CPU val]", leave=True)
    with torch.no_grad():
        for batch in pbar:
            input_data = batch['input'].to(device)
            target_data = batch['target'].to(device)
            max_dose = batch['max_dose'].to(device)

            max_dose_view = max_dose.view(-1, 1, 1, 1, 1) + 1e-8
            input_data = input_data / max_dose_view
            target_data = target_data / max_dose_view

            output = model(input_data)
            loss, _ = loss_fn(output, target_data, max_dose.max())
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.6f}')

    avg_loss = total_loss / len(dataloader)
    print(f"✓ Val Loss: {avg_loss:.6f}")
    return avg_loss

# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n[1/4] Loading datasets...")
    train_dataset = SimpleDoseDatasetWeighted(TRAIN_DIR, DATASET_ROOT, INPUT_LEVELS, split='train')
    val_dataset = SimpleDoseDatasetWeighted(VAL_DIR, DATASET_ROOT, INPUT_LEVELS, split='val')

    # Sampler con solo 50 iteraciones por época (benchmark rápido)
    sampler = WeightedRandomSampler(
        torch.tensor(train_dataset.dose_weights),
        num_samples=200,  # Igual que GPU (800 samples / batch 4 = 200 iters)
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[2/4] Building model on CPU...")
    model = UNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DynamicDoseLoss(
        threshold_core=DOSE_THRESHOLD_CORE,
        threshold_low=DOSE_THRESHOLD_SIGNIFICANT,
        weight_min=0.1
    )

    print(f"\n[3/4] Starting {NUM_EPOCHS}-epoch CPU benchmark...")
    times = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} [CPU]")
        print(f"{'='*70}")

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch)
        t1 = time.time()
        epoch_time = t1 - t0
        times.append(epoch_time)

        val_loss = validate(model, val_loader, loss_fn, DEVICE, epoch)
        print(f"  ⏱ Epoch time: {epoch_time:.1f}s ({epoch_time/200:.3f} s/it)")

    # Resumen
    print(f"\n{'='*70}")
    print(f"CPU BENCHMARK RESULTS")
    print(f"{'='*70}")
    avg_time = sum(times) / len(times)
    avg_sit = avg_time / 200
    print(f"  Average epoch time: {avg_time:.1f}s")
    print(f"  Average s/it:       {avg_sit:.3f}")
    print(f"  200 iterations/epoch (batch={BATCH_SIZE})")
    print(f"\n  Para comparar con GPU (cudnn off): ~1.07 s/it")
    print(f"  Ratio CPU/GPU:      {avg_sit/1.07:.2f}x")

    # Guardar resultados
    results = {
        'device': 'cpu',
        'batch_size': BATCH_SIZE,
        'base_channels': BASE_CHANNELS,
        'epoch_times': times,
        'avg_s_per_it': avg_sit,
        'gpu_s_per_it_reference': 1.07,
        'ratio': avg_sit / 1.07
    }
    with open(RESULTS_DIR / 'cpu_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_DIR / 'cpu_benchmark.json'}")

if __name__ == '__main__':
    main()
