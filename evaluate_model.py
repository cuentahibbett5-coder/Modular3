#!/usr/bin/env python3
"""
Evaluación del modelo entrenado: best_model.pt
- Carga volúmenes de validación completos
- Inferencia por sliding window (parches de 96³ con overlap)
- Métricas: PSNR, SSIM, Error Relativo por zona de dosis
- Genera imágenes de comparación: Input | Predicción | GT | Error
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Sin display en cluster
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

torch.backends.cudnn.enabled = False

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 16
MODEL_PATH = Path("runs/denoising_v2_residual/best_model.pt")
DATASET_ROOT = Path("dataset_pilot")
VAL_DIR = DATASET_ROOT / "val"
EVAL_DIR = Path("runs/denoising_v2_residual/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]
PATCH_SIZE = 96
OVERLAP = 16  # Overlap entre parches para evitar artefactos de borde

print(f"✓ Device: {DEVICE}")
print(f"✓ Model: {MODEL_PATH}")
print(f"✓ Output: {EVAL_DIR}")

# ============================================================================
# 3D U-NET RESIDUAL (idéntica al training v2)
# ============================================================================
class ResidualUNet3D(nn.Module):
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
        return x + residual  # RESIDUAL

# ============================================================================
# SLIDING WINDOW INFERENCE
# ============================================================================
def sliding_window_inference(model, volume, device, patch_size=96, overlap=16):
    """
    Inferencia por ventana deslizante con overlap para volúmenes completos.
    Evita OOM al no procesar todo el volumen de golpe.
    """
    model.eval()
    z, y, x = volume.shape
    step = patch_size - overlap
    
    # Pad volume to be divisible by step
    pad_z = (step - (z % step)) % step
    pad_y = (step - (y % step)) % step
    pad_x = (step - (x % step)) % step
    
    padded = np.pad(volume, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
    pz, py, px = padded.shape
    
    output = np.zeros_like(padded)
    weight_map = np.zeros_like(padded)
    
    # Ventana de ponderación (más peso al centro, menos al borde)
    w = np.ones((patch_size, patch_size, patch_size), dtype=np.float32)
    margin = overlap // 2
    if margin > 0:
        for i in range(margin):
            fade = (i + 1) / margin
            w[i, :, :] *= fade
            w[-(i+1), :, :] *= fade
            w[:, i, :] *= fade
            w[:, -(i+1), :] *= fade
            w[:, :, i] *= fade
            w[:, :, -(i+1)] *= fade
    
    positions = []
    
    # Generate all positions with proper edge handling
    for zs in range(0, pz, step):
        for ys in range(0, py, step):
            for xs in range(0, px, step):
                # Ensure we don't go beyond volume bounds
                z_end = min(zs + patch_size, pz)
                y_end = min(ys + patch_size, py)
                x_end = min(xs + patch_size, px)
                
                # Adjust start if patch would be too small at edge
                if z_end - zs < patch_size:
                    zs = max(0, z_end - patch_size)
                if y_end - ys < patch_size:
                    ys = max(0, y_end - patch_size)  
                if x_end - xs < patch_size:
                    xs = max(0, x_end - patch_size)
                
                positions.append((zs, ys, xs))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)
    positions = unique_positions
    
    with torch.no_grad():
        for zs, ys, xs in tqdm(positions, desc="  Inference", leave=False):
            patch = padded[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size]
            patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            
            pred = model(patch_t)
            pred_np = pred.squeeze().cpu().numpy()
            
            output[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += pred_np * w
            weight_map[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += w
    
    # Normalizar por pesos
    weight_map = np.maximum(weight_map, 1e-8)
    output = output / weight_map
    
    # Quitar padding
    return output[:z, :y, :x]

# ============================================================================
# MÉTRICAS
# ============================================================================
def calc_psnr(pred, target, max_val=None):
    """Peak Signal-to-Noise Ratio"""
    if max_val is None:
        max_val = target.max()
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-15:
        return 100.0
    return 10 * np.log10(max_val ** 2 / mse)

def calc_ssim_3d(pred, target, window_size=7):
    """Simplified 3D SSIM (per-slice mean)"""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * target.max()) ** 2
    C2 = (0.03 * target.max()) ** 2
    
    mu_p = uniform_filter(pred, size=window_size)
    mu_t = uniform_filter(target, size=window_size)
    
    sig_pp = uniform_filter(pred ** 2, size=window_size) - mu_p ** 2
    sig_tt = uniform_filter(target ** 2, size=window_size) - mu_t ** 2
    sig_pt = uniform_filter(pred * target, size=window_size) - mu_p * mu_t
    
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_pp + sig_tt + C2))
    
    return ssim_map.mean()

def calc_dose_metrics(pred, target, max_dose):
    """Métricas por zona de dosis"""
    results = {}
    
    zones = {
        'high_dose (≥20%)': target >= 0.20 * max_dose,
        'mid_dose (1-20%)': (target >= 0.01 * max_dose) & (target < 0.20 * max_dose),
        'low_dose (<1%)': (target < 0.01 * max_dose) & (target > 0),
    }
    
    for name, mask in zones.items():
        n_voxels = mask.sum()
        if n_voxels == 0:
            results[name] = {'n_voxels': 0, 'mae': 0, 'rel_error_%': 0}
            continue
        
        mae = np.abs(pred[mask] - target[mask]).mean()
        rel_error = np.abs(pred[mask] - target[mask]) / (np.abs(target[mask]) + 1e-10)
        
        results[name] = {
            'n_voxels': int(n_voxels),
            'mae': float(mae),
            'rel_error_%': float(rel_error.mean() * 100),
            'max_error_%': float(rel_error.max() * 100),
            'psnr': float(calc_psnr(pred[mask], target[mask], max_dose))
        }
    
    return results

def calc_advanced_metrics(pred, target, max_dose):
    """Métricas avanzadas para evaluación de dosis"""
    results = {}
    
    # 1. Normalized Cross-Correlation (NCC)
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    ncc = np.corrcoef(pred_flat, target_flat)[0, 1]
    results['ncc'] = float(ncc) if not np.isnan(ncc) else 0.0
    
    # 2. Mean Dose comparison
    mean_pred = np.mean(pred)
    mean_target = np.mean(target)
    mean_dose_error = abs(mean_pred - mean_target) / mean_target * 100 if mean_target > 0 else 0
    results['mean_dose_error_%'] = float(mean_dose_error)
    
    # 3. Max Dose comparison
    max_pred = np.max(pred)
    max_target = np.max(target)
    max_dose_error = abs(max_pred - max_target) / max_target * 100 if max_target > 0 else 0
    results['max_dose_error_%'] = float(max_dose_error)
    
    # 4. PDD Curve similarity (depth dose)
    z_size = pred.shape[0]
    pdd_pred = np.array([pred[z].max() for z in range(z_size)])
    pdd_target = np.array([target[z].max() for z in range(z_size)])
    
    pdd_mae = np.mean(np.abs(pdd_pred - pdd_target))
    pdd_rel_error = np.mean(np.abs(pdd_pred - pdd_target) / (pdd_target + 1e-10)) * 100
    
    results['pdd_mae'] = float(pdd_mae)
    results['pdd_rel_error_%'] = float(pdd_rel_error)
    results['pdd_corr'] = float(np.corrcoef(pdd_pred, pdd_target)[0, 1])
    
    # 5. Isodose volume comparison (volúmenes a diferentes niveles)
    isodose_levels = [0.95, 0.90, 0.80, 0.50, 0.20]  # 95%, 90%, etc. of max dose
    isodose_metrics = {}
    
    for level in isodose_levels:
        threshold = level * max_dose
        vol_pred = np.sum(pred >= threshold)
        vol_target = np.sum(target >= threshold)
        
        if vol_target > 0:
            vol_error = abs(vol_pred - vol_target) / vol_target * 100
            isodose_metrics[f'V{int(level*100)}%'] = {
                'pred_voxels': int(vol_pred),
                'target_voxels': int(vol_target),
                'volume_error_%': float(vol_error)
            }
    
    results['isodose_volumes'] = isodose_metrics
    
    # 6. Gamma Analysis (simplified 3D)
    mask_significant = target > 0.10 * max_dose  # Solo en región de alta dosis
    if mask_significant.sum() > 0:
        gamma_pass_rate = calc_gamma_pass_rate(pred, target, mask_significant)
        results['gamma_pass_rate_%'] = float(gamma_pass_rate)
    else:
        results['gamma_pass_rate_%'] = 0.0
    
    return results

def calc_gamma_pass_rate(pred, target, mask, dose_tolerance=3.0, distance_tolerance=3.0):
    """
    Simplified Gamma Analysis - % of voxels passing gamma < 1
    dose_tolerance: % de tolerancia en dosis
    distance_tolerance: mm de tolerancia espacial (asumimos voxels de 1mm)
    """
    pred_masked = pred[mask]
    target_masked = target[mask]
    
    # Diferencia de dosis normalizada
    dose_diff = np.abs(pred_masked - target_masked)
    dose_gamma = dose_diff / (dose_tolerance / 100.0 * target_masked.max())
    
    # Simplificación: Solo gamma de dosis (sin distancia espacial completa)
    # Para gamma completo necesitaríamos calcular distancias 3D
    gamma_values = dose_gamma  # Versión simplificada
    
    pass_rate = np.sum(gamma_values <= 1.0) / len(gamma_values) * 100
    return pass_rate

# ============================================================================
# VISUALIZACIÓN
# ============================================================================
def plot_comparison(input_vol, pred_vol, target_vol, z_slice, save_path, pair_name, level):
    """4 paneles: Input | Predicción | GT | Error Relativo"""
    # DEBUG: Verificar dimensiones de los slices
    print(f"      DEBUG plot_comparison - slice {z_slice}:")
    print(f"        input_vol[{z_slice}].shape = {input_vol[z_slice].shape}")
    print(f"        pred_vol[{z_slice}].shape = {pred_vol[z_slice].shape}")
    print(f"        target_vol[{z_slice}].shape = {target_vol[z_slice].shape}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    vmax = target_vol[z_slice].max()
    vmin = 0
    
    # Row 1: Volúmenes
    im0 = axes[0, 0].imshow(input_vol[z_slice], cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'Input ({level})', fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(pred_vol[z_slice], cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title('Predicción (U-Net)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(target_vol[z_slice], cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 2].set_title('Ground Truth (29.4M)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: Errores
    diff_input = np.abs(input_vol[z_slice] - target_vol[z_slice])
    diff_pred = np.abs(pred_vol[z_slice] - target_vol[z_slice])
    diff_max = max(diff_input.max(), diff_pred.max(), 1e-10)
    
    im3 = axes[1, 0].imshow(diff_input, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1, 0].set_title(f'|Input - GT|', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    im4 = axes[1, 1].imshow(diff_pred, cmap='viridis', vmin=0, vmax=diff_max, aspect='auto')
    axes[1, 1].set_title(f'|Pred - GT|', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    # Error relativo (solo donde GT > 1% del max)
    mask = target_vol[z_slice] > 0.01 * vmax
    rel_err = np.zeros_like(target_vol[z_slice])
    if mask.any():
        rel_err[mask] = np.abs(pred_vol[z_slice][mask] - target_vol[z_slice][mask]) / target_vol[z_slice][mask] * 100
    
    im5 = axes[1, 2].imshow(rel_err, cmap='RdYlGn_r', vmin=0, vmax=20, aspect='auto')
    axes[1, 2].set_title('Error Relativo % (pred vs GT)', fontsize=12)
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, label='%')
    
    for ax in axes.flat:
        ax.axis('off')
    
    fig.suptitle(f'{pair_name} | {level} | z={z_slice}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_pdd_comparison(input_vol, pred_vol, target_vol, save_path, pair_name, level):
    """PDD: Max dose por capa z → Input vs Pred vs GT"""
    z_size = target_vol.shape[0]
    
    pdd_input = np.array([input_vol[z].max() for z in range(z_size)])
    pdd_pred = np.array([pred_vol[z].max() for z in range(z_size)])
    pdd_gt = np.array([target_vol[z].max() for z in range(z_size)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDD absoluto
    axes[0].plot(pdd_gt, 'k-', linewidth=2, label='GT (29.4M)', alpha=0.8)
    axes[0].plot(pdd_input, 'r--', linewidth=1, label=f'Input ({level})', alpha=0.6)
    axes[0].plot(pdd_pred, 'b-', linewidth=1.5, label='Predicción', alpha=0.8)
    axes[0].set_xlabel('Capa Z')
    axes[0].set_ylabel('Max Dose')
    axes[0].set_title('PDD (Percent Depth Dose)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error relativo del PDD
    mask = pdd_gt > 0.01 * pdd_gt.max()
    rel_input = np.zeros_like(pdd_gt)
    rel_pred = np.zeros_like(pdd_gt)
    rel_input[mask] = np.abs(pdd_input[mask] - pdd_gt[mask]) / pdd_gt[mask] * 100
    rel_pred[mask] = np.abs(pdd_pred[mask] - pdd_gt[mask]) / pdd_gt[mask] * 100
    
    axes[1].plot(rel_input, 'r--', linewidth=1, label=f'Input ({level})', alpha=0.6)
    axes[1].plot(rel_pred, 'b-', linewidth=1.5, label='Predicción', alpha=0.8)
    axes[1].set_xlabel('Capa Z')
    axes[1].set_ylabel('Error Relativo (%)')
    axes[1].set_title('PDD Error Relativo')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, min(100, max(rel_input[mask].max(), rel_pred[mask].max()) * 1.2) if mask.any() else 100)
    
    fig.suptitle(f'{pair_name} | {level}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_input_vs_pred_quality(input_vol, pred_vol, target_vol, pair_name, level, eval_dir):
    """Gráficas comparativas específicas: Input vs Predicción"""
    
    # 1. MEJORA EN ERRORES PDD
    z_size = target_vol.shape[0]
    pdd_input = np.array([input_vol[z].max() for z in range(z_size)])
    pdd_pred = np.array([pred_vol[z].max() for z in range(z_size)])
    pdd_target = np.array([target_vol[z].max() for z in range(z_size)])
    
    err_input = np.abs(pdd_input - pdd_target) / (pdd_target + 1e-8) * 100
    err_pred = np.abs(pdd_pred - pdd_target) / (pdd_target + 1e-8) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Curvas PDD
    axes[0].plot(pdd_target, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[0].plot(pdd_input, 'r--', linewidth=2, label=f'Input ({level})', alpha=0.7)
    axes[0].plot(pdd_pred, 'b-', linewidth=2, label='Predicción IA', alpha=0.8)
    axes[0].set_xlabel('Profundidad Z')
    axes[0].set_ylabel('Dosis Máxima')
    axes[0].set_title('Curvas PDD Comparativas')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Errores comparativos
    axes[1].plot(err_input, 'r--', linewidth=2, label='Error Input', alpha=0.7)
    axes[1].plot(err_pred, 'b-', linewidth=2, label='Error Predicción', alpha=0.8)
    axes[1].set_xlabel('Profundidad Z')
    axes[1].set_ylabel('Error Relativo (%)')
    axes[1].set_title('Errores PDD Comparativos')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Estadísticas
    input_mae = np.mean(err_input)
    pred_mae = np.mean(err_pred)
    improvement = input_mae / pred_mae if pred_mae > 0 else 1
    
    fig.suptitle(f'{pair_name} {level} - PDD Quality Analysis\n'
                f'Error promedio: Input={input_mae:.1f}%, IA={pred_mae:.1f}% (mejora {improvement:.1f}×)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(eval_dir / f'{pair_name}_{level}_pdd_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. CORRELACIÓN SCATTER
    mask = target_vol > 0.05 * target_vol.max()
    target_flat = target_vol[mask]
    input_flat = input_vol[mask]
    pred_flat = pred_vol[mask]
    
    # Submuestreo para visualización
    n_sample = min(5000, len(target_flat))
    if n_sample > 0:
        idx = np.random.choice(len(target_flat), n_sample, replace=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Input vs Target
        axes[0].scatter(target_flat[idx], input_flat[idx], alpha=0.6, s=2, c='red')
        axes[0].plot([0, target_vol.max()], [0, target_vol.max()], 'k--', alpha=0.8)
        corr_input = np.corrcoef(target_flat, input_flat)[0, 1]
        axes[0].set_xlabel('Ground Truth')
        axes[0].set_ylabel('Input')  
        axes[0].set_title(f'Input vs GT (r={corr_input:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Pred vs Target  
        axes[1].scatter(target_flat[idx], pred_flat[idx], alpha=0.6, s=2, c='blue')
        axes[1].plot([0, target_vol.max()], [0, target_vol.max()], 'k--', alpha=0.8)
        corr_pred = np.corrcoef(target_flat, pred_flat)[0, 1]
        axes[1].set_xlabel('Ground Truth')
        axes[1].set_ylabel('Predicción')
        axes[1].set_title(f'Predicción vs GT (r={corr_pred:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{pair_name} {level} - Correlation Analysis\n'
                    f'Mejora correlación: r={corr_input:.4f} → r={corr_pred:.4f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(eval_dir / f'{pair_name}_{level}_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"      ✓ Quality comparison plots saved")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("EVALUACIÓN: best_model.pt")
    print("=" * 70)
    
    # Cargar modelo
    print("\n[1/3] Cargando modelo...")
    model = ResidualUNet3D(base_channels=BASE_CHANNELS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"  ✓ Modelo cargado (época {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.6f})")
    
    # Buscar pares de validación
    print("\n[2/3] Cargando datos de validación...")
    n_targets = len(list(DATASET_ROOT.glob("target_*")))
    pair_dirs = sorted(VAL_DIR.glob("pair_*"))
    print(f"  Found {len(pair_dirs)} validation pairs")
    
    all_metrics = {}
    
    # Evaluar cada par y nivel
    print("\n[3/3] Evaluando...")
    for pair_dir in pair_dirs:
        pair_num = int(pair_dir.name.split("_")[-1])
        target_idx = ((pair_num - 1) % n_targets) + 1
        target_mhd = DATASET_ROOT / f"target_{target_idx}" / "dose_edep.mhd"
        
        if not target_mhd.exists():
            print(f"  ⚠ Target missing: {target_mhd}")
            continue
        
        target_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(target_mhd))).astype(np.float32)
        max_dose = target_vol.max()
        print(f"\n  {pair_dir.name} (target_{target_idx}, max_dose={max_dose:.4f})")
        
        for level in INPUT_LEVELS:
            input_mhd = pair_dir / f"{level}.mhd"
            if not input_mhd.exists():
                input_mhd = pair_dir / level / "dose_edep.mhd"
            if not input_mhd.exists():
                continue
            
            input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_mhd))).astype(np.float32)
            
            # Normalizar → inferencia → desnormalizar
            input_norm = input_vol / (max_dose + 1e-8)
            print(f"    {level}: inferencia sliding window...", end=" ", flush=True)
            pred_norm = sliding_window_inference(model, input_norm, DEVICE, PATCH_SIZE, OVERLAP)
            pred_vol = pred_norm * max_dose  # Desnormalizar
            
            # DEBUG: Verificar dimensiones
            print(f"\n      DEBUG - Shapes: input={input_vol.shape}, pred={pred_vol.shape}, target={target_vol.shape}")
            
            # Métricas globales
            psnr_input = calc_psnr(input_vol, target_vol, max_dose)
            psnr_pred = calc_psnr(pred_vol, target_vol, max_dose)
            
            try:
                ssim_pred = calc_ssim_3d(pred_vol, target_vol)
            except ImportError:
                ssim_pred = -1  # scipy no disponible
            
            # Métricas por zona de dosis
            dose_metrics = calc_dose_metrics(pred_vol, target_vol, max_dose)
            
            # Métricas avanzadas
            advanced_metrics = calc_advanced_metrics(pred_vol, target_vol, max_dose)
            
            key = f"{pair_dir.name}_{level}"
            all_metrics[key] = {
                'psnr_input': float(psnr_input),
                'psnr_pred': float(psnr_pred),
                'psnr_gain_dB': float(psnr_pred - psnr_input),
                'ssim_pred': float(ssim_pred),
                'dose_zones': dose_metrics,
                'advanced': advanced_metrics
            }
            
            print(f"PSNR: {psnr_input:.1f} → {psnr_pred:.1f} dB (gain: +{psnr_pred-psnr_input:.1f} dB)")
            print(f"SSIM: {ssim_pred:.4f}")
            
            # Visualizaciones
            # DEBUG: Verificar dimensiones ANTES de plot_comparison
            print(f"\n      DEBUG PRE-PLOT:")
            print(f"        input_vol.shape = {input_vol.shape}")
            print(f"        pred_vol.shape = {pred_vol.shape}")
            print(f"        target_vol.shape = {target_vol.shape}")
            print(f"        max values: input={input_vol.max():.4f}, pred={pred_vol.max():.4f}, target={target_vol.max():.4f}")
            
            # Slice central (core) 
            z_core = target_vol.shape[0] // 2
            # Buscar slice con más dosis
            pdd = np.array([target_vol[z].max() for z in range(target_vol.shape[0])])
            z_max = int(np.argmax(pdd))
            print(f"        z_core = {z_core}, z_max = {z_max}")
            
            for z_slice, z_name in [(z_core, "core"), (z_max, "peak")]:
                save_path = EVAL_DIR / f"{key}_z{z_slice}_{z_name}.png"
                print(f"        Plotting slice {z_slice} ({z_name})...")
                plot_comparison(input_vol, pred_vol, target_vol, z_slice, save_path, pair_dir.name, level)
            
            # PDD
            pdd_path = EVAL_DIR / f"{key}_pdd.png"
            plot_pdd_comparison(input_vol, pred_vol, target_vol, pdd_path, pair_dir.name, level)
            
            # Quality comparison plots
            plot_input_vs_pred_quality(input_vol, pred_vol, target_vol, pair_dir.name, level, EVAL_DIR)
    
    # Guardar métricas
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 70)
    
    psnr_gains = []
    for key, m in all_metrics.items():
        print(f"\n  {key}:")
        print(f"    PSNR:  {m['psnr_input']:.1f} → {m['psnr_pred']:.1f} dB  (gain: +{m['psnr_gain_dB']:.1f} dB)")
        if m['ssim_pred'] > 0:
            print(f"    SSIM:  {m['ssim_pred']:.4f}")
        
        # Métricas avanzadas
        adv = m['advanced']
        print(f"    NCC:   {adv['ncc']:.4f}")
        print(f"    Mean dose error: {adv['mean_dose_error_%']:.1f}%")
        print(f"    Max dose error:  {adv['max_dose_error_%']:.1f}%")
        print(f"    PDD correlation: {adv['pdd_corr']:.4f}")
        if adv['gamma_pass_rate_%'] > 0:
            print(f"    Gamma pass rate: {adv['gamma_pass_rate_%']:.1f}%")
        
        psnr_gains.append(m['psnr_gain_dB'])
        
        for zone, zm in m['dose_zones'].items():
            if zm['n_voxels'] > 0:
                print(f"    {zone}: error={zm['rel_error_%']:.1f}% ({zm['n_voxels']:,} voxels)")
    
    avg_gain = np.mean(psnr_gains)
    print(f"\n  PSNR gain promedio: +{avg_gain:.1f} dB")
    print(f"\n  Imágenes guardadas en: {EVAL_DIR}")
    print(f"  Métricas guardadas en: {metrics_path}")

if __name__ == '__main__':
    main()
