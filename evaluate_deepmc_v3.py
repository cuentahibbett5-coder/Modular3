#!/usr/bin/env python3
"""
Evaluación del modelo DeepMC v3
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Import del modelo
sys.path.insert(0, str(Path(__file__).parent))
from train_deepmc_v3 import DeepMCNet

torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path("runs/denoising_deepmc_v3/best_model.pt")
EVAL_DIR = Path("runs/denoising_deepmc_v3/evaluation")
VAL_DATA_DIR = Path("dataset_pilot/val")

EVAL_DIR.mkdir(parents=True, exist_ok=True)

def load_model():
    model = DeepMCNet(base_channels=16, dual_input=False).to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"✓ Loaded model from {MODEL_PATH}")
    else:
        logger.error(f"✗ Model not found at {MODEL_PATH}")
        return None
    model.eval()
    return model

def sliding_window_inference(model, volume, patch_size=96, overlap=16):
    """
    Realiza inferencia con ventana deslizante para volúmenes grandes.
    """
    z, y, x = volume.shape
    output = np.zeros_like(volume)
    count = np.zeros_like(volume, dtype=np.float32)
    
    stride = patch_size - overlap
    
    for pz in range(0, z - patch_size + 1, stride):
        for py in range(0, y - patch_size + 1, stride):
            for px in range(0, x - patch_size + 1, stride):
                patch = volume[pz:pz+patch_size, py:py+patch_size, px:px+patch_size]
                
                # Convertir a tensor
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    pred_patch = model(patch_tensor, ct=None).cpu().numpy()
                
                pred_patch = pred_patch[0, 0]  # Remove batch y channel dims
                
                output[pz:pz+patch_size, py:py+patch_size, px:px+patch_size] += pred_patch
                count[pz:pz+patch_size, py:py+patch_size, px:px+patch_size] += 1
    
    # Normalizar por superposición
    output = np.divide(output, count, where=count > 0, out=output)
    
    return output

def psnr(img1, img2, max_val=None):
    """Peak Signal-to-Noise Ratio"""
    if max_val is None:
        max_val = img2.max()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def ssim_3d(img1, img2, win_size=11):
    """Simplified 3D SSIM"""
    from scipy.ndimage import gaussian_filter
    
    mu1 = gaussian_filter(img1.astype(float), sigma=1)
    mu2 = gaussian_filter(img2.astype(float), sigma=1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(img1.astype(float) ** 2, sigma=1) - mu1_sq
    sigma2_sq = gaussian_filter(img2.astype(float) ** 2, sigma=1) - mu2_sq
    sigma12 = gaussian_filter(img1.astype(float) * img2, sigma=1) - mu1_mu2
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim_map.mean()

def evaluate():
    model = load_model()
    if model is None:
        return
    
    logger.info(f"Evaluating on validation set...")
    
    # Iterar sobre pacientes de validación
    for patient_dir in sorted(VAL_DATA_DIR.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        gt_file = patient_dir / "gt.nii.gz"
        input_file = patient_dir / "input_10M.nii.gz"
        
        if not (gt_file.exists() and input_file.exists()):
            continue
        
        patient_name = patient_dir.name
        logger.info(f"\n  Processing {patient_name}...")
        
        # Cargar volúmenes
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file))).astype(np.float32)
        input_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(input_file))).astype(np.float32)
        
        # Inferencia
        with torch.no_grad():
            pred_vol = sliding_window_inference(model, input_vol, patch_size=96, overlap=16)
        
        # Métricas
        max_dose = gt_vol.max()
        psnr_val = psnr(pred_vol, gt_vol, max_val=max_dose)
        ssim_val = ssim_3d(pred_vol, gt_vol)
        
        logger.info(f"    PSNR: {psnr_val:.2f} dB")
        logger.info(f"    SSIM: {ssim_val:.4f}")
        
        # Análisis por zona de dosis
        high_dose_mask = (gt_vol >= 0.8 * max_dose)
        mid_dose_mask = (gt_vol >= 0.2 * max_dose) & (gt_vol < 0.8 * max_dose)
        low_dose_mask = (gt_vol > 0) & (gt_vol < 0.2 * max_dose)
        
        if high_dose_mask.sum() > 0:
            error_high = np.mean(np.abs(pred_vol[high_dose_mask] - gt_vol[high_dose_mask]))
            logger.info(f"    High Dose Error: {error_high:.6f}")
        
        if mid_dose_mask.sum() > 0:
            error_mid = np.mean(np.abs(pred_vol[mid_dose_mask] - gt_vol[mid_dose_mask]))
            logger.info(f"    Mid Dose Error: {error_mid:.6f}")
        
        if low_dose_mask.sum() > 0:
            error_low = np.mean(np.abs(pred_vol[low_dose_mask] - gt_vol[low_dose_mask]))
            logger.info(f"    Low Dose Error: {error_low:.6f}")
        
        # Guardar PDD
        pred_pdd = pred_vol.sum(axis=(1, 2)) / (pred_vol.shape[1] * pred_vol.shape[2])
        gt_pdd = gt_vol.sum(axis=(1, 2)) / (gt_vol.shape[1] * gt_vol.shape[2])
        
        np.save(EVAL_DIR / f"{patient_name}_pred_pdd.npy", pred_pdd)
        np.save(EVAL_DIR / f"{patient_name}_gt_pdd.npy", gt_pdd)
        
        logger.info(f"    ✓ Saved PDD for {patient_name}")

if __name__ == "__main__":
    evaluate()
