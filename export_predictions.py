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
# MODEL (same as evaluate_model.py)
# ============================================================================
class UNet3D(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.upconv3 = torch.nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = torch.nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = torch.nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        self.out_conv = torch.nn.Conv3d(base_channels, out_channels, 1)
        
        self.pool = torch.nn.MaxPool3d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm3d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        input_x = x
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
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
        
        residual = self.out_conv(d1)
        output = input_x + residual
        
        return output

def sliding_window_inference(model, volume, patch_size=96, overlap=16):
    """Sliding window inference with overlap averaging"""
    device = next(model.parameters()).device
    D, H, W = volume.shape
    stride = patch_size - overlap
    
    output = np.zeros_like(volume)
    counts = np.zeros_like(volume)
    
    for z in range(0, D - patch_size + 1, stride):
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = volume[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    pred_patch = model(patch_tensor).squeeze().cpu().numpy()
                
                output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred_patch
                counts[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
    
    output = output / np.maximum(counts, 1)
    return output

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
    model = UNet3D(base_channels=16).to(device)
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
        
        # Load target
        target_path = list(pair_dir.glob("target_*.mhd"))[0]
        target_sitk = sitk.ReadImage(str(target_path))
        target_vol = sitk.GetArrayFromImage(target_sitk)
        max_dose = target_vol.max()
        
        print(f"\n  {pair_name} (max_dose={max_dose:.1f})")
        
        # Process each input level
        input_files = sorted(pair_dir.glob("input_*.mhd"))
        for input_path in input_files:
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
