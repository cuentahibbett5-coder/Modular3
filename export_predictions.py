#!/usr/bin/env python3
"""
Export raw prediction data for local analysis
Saves: input, prediction, target volumes as .npy files
"""
import os
import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "runs/denoising_v2_residual/best_model.pt"
DATA_DIR = Path("dataset_pilot/val")
OUTPUT_DIR = Path("exports")
PATCH_SIZE = 96
OVERLAP = 16

# ============================================================================
# MODEL (same as train_v2.py)
# ============================================================================
class ResidualUNet3D(torch.nn.Module):
    """U-Net que predice el RESIDUAL (corrección) en vez de la dosis absoluta."""
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = torch.nn.MaxPool3d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = torch.nn.MaxPool3d(2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = torch.nn.MaxPool3d(2)
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        self.upconv3 = torch.nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = torch.nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = torch.nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        self.final = torch.nn.Conv3d(base_channels, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True)
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
        return x + residual

def sliding_window_inference(model, volume, patch_size=96, overlap=16):
    """
    Sliding window inference with overlap weighting (same as evaluate_model.py)
    Better quality than simple averaging at patch boundaries.
    """
    device = next(model.parameters()).device
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
    
    # Weight window (more weight at center, less at edges)
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
    
    with torch.no_grad():
        for zs in range(0, pz - patch_size + 1, step):
            for ys in range(0, py - patch_size + 1, step):
                for xs in range(0, px - patch_size + 1, step):
                    patch = padded[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size]
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    pred_patch = model(patch_tensor).squeeze().cpu().numpy()
                    
                    output[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += pred_patch * w
                    weight_map[zs:zs+patch_size, ys:ys+patch_size, xs:xs+patch_size] += w
    
    # Normalize by weights
    weight_map = np.maximum(weight_map, 1e-8)
    output = output / weight_map
    
    # Remove padding
    return output[:z, :y, :x]

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("EXPORTANDO PREDICCIONES")
    print("="*70)
    
    # Load model
    print("\n[1/3] Cargando modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResidualUNet3D(base_channels=16).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            epoch = checkpoint.get('epoch', '?')
            print(f"  ✓ Modelo cargado (época {epoch})")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', '?')
            print(f"  ✓ Modelo cargado (época {epoch})")
        else:
            model.load_state_dict(checkpoint)
            print(f"  ✓ Modelo cargado")
    else:
        model.load_state_dict(checkpoint)
        print(f"  ✓ Modelo cargado")
    
    model.eval()
    
    # Create output dir
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find validation pairs
    print("\n[2/3] Buscando pares de validación...")
    pairs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"  Found {len(pairs)} validation pairs")
    
    # Process each pair
    print("\n[3/3] Exportando...")
    for pair_dir in pairs:
        pair_name = pair_dir.name
        
        # Load target (find .mhd file that's not an input)
        all_mhd = list(pair_dir.glob("*.mhd"))
        input_files = [f for f in all_mhd if 'input' in f.name.lower()]
        target_files = [f for f in all_mhd if 'input' not in f.name.lower()]
        
        if not target_files:
            print(f"\n  ⚠ Skipping {pair_name}: no target file found")
            continue
        
        target_path = target_files[0]
        target_sitk = sitk.ReadImage(str(target_path))
        target_vol = sitk.GetArrayFromImage(target_sitk)
        max_dose = target_vol.max()
        
        print(f"\n  {pair_name} (max_dose={max_dose:.1f})")
        
        # Process each input level
        if not input_files:
            print(f"    ⚠ No input files found")
            continue
            
        for input_path in sorted(input_files):
            level = input_path.stem.split('_')[-1]  # e.g., "1M", "2M"
            
            # Load input
            input_sitk = sitk.ReadImage(str(input_path))
            input_vol = sitk.GetArrayFromImage(input_sitk)
            
            # Predict
            pred_vol = sliding_window_inference(model, input_vol, PATCH_SIZE, OVERLAP)
            
            # Save as .npy
            output_name = f"{pair_name}_{level}"
            np.save(OUTPUT_DIR / f"{output_name}_input.npy", input_vol)
            np.save(OUTPUT_DIR / f"{output_name}_pred.npy", pred_vol)
            np.save(OUTPUT_DIR / f"{output_name}_target.npy", target_vol)
            
            print(f"    ✓ {level}: {input_vol.shape} → exports/{output_name}_*.npy")
    
    print("\n" + "="*70)
    print(f"EXPORTACIÓN COMPLETA: {OUTPUT_DIR}/")
    print("="*70)
    print("\nArchivos .npy guardados:")
    print(f"  - *_input.npy   (input ruidoso)")
    print(f"  - *_pred.npy    (predicción del modelo)")
    print(f"  - *_target.npy  (ground truth)")
    print("\nPara cargar localmente:")
    print("  import numpy as np")
    print("  data = np.load('exports/pair_021_1M_pred.npy')")

if __name__ == "__main__":
    main()
