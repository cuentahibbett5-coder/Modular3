#!/usr/bin/env python3
"""
Analyze model predictions layer-by-layer across the volume.
Shows MAE, RMSE, correlation, and scaling factor for each z-slice.
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import argparse

def get_best_model_path():
    """Auto-detect best model from fullvol or v2 runs"""
    runs_dir = Path('runs')
    if not runs_dir.exists():
        raise FileNotFoundError("runs/ directory not found")
    
    best_models = []
    
    # Look for denoising_fullvol or denoising_v2 directories
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            best_model = run_dir / 'best.pt'
            if best_model.exists():
                best_models.append(best_model)
    
    if not best_models:
        raise FileNotFoundError("No best.pt models found in runs/")
    
    # Return the most recently modified
    return max(best_models, key=lambda p: p.stat().st_mtime)

def load_sample(idx, data_dir='data'):
    """Load a specific training sample"""
    inp_path = Path(data_dir) / f'input_{idx:03d}.npy'
    tgt_path = Path(data_dir) / f'target_{idx:03d}.npy'
    
    if not inp_path.exists() or not tgt_path.exists():
        return None, None, None
    
    inp = np.load(inp_path)
    tgt = np.load(tgt_path)
    
    return inp, tgt, f"Sample {idx:03d}"

def normalize_sample(inp, tgt):
    """Normalize using max(input) like training does"""
    max_inp = float(np.max(inp))
    if max_inp > 0:
        inp_norm = inp / max_inp
        tgt_norm = tgt / max_inp
    else:
        inp_norm = inp
        tgt_norm = tgt
    return inp_norm, tgt_norm

def analyze_layers(target, prediction, num_layers=None):
    """
    Analyze each layer (z-slice) and return statistics
    
    Args:
        target: (D, H, W) target volume
        prediction: (D, H, W) predicted volume
        num_layers: number of layers to analyze (default: all)
    
    Returns:
        dict with per-layer statistics
    """
    D = target.shape[0]
    if num_layers is None:
        num_layers = D
    
    layers = np.linspace(0, D-1, num_layers, dtype=int)
    
    stats_list = []
    
    for layer_idx in layers:
        tgt_slice = target[layer_idx].flatten()
        pred_slice = prediction[layer_idx].flatten()
        
        # Skip if slice is mostly zeros
        if np.sum(tgt_slice) < 1e-6:
            continue
        
        mae = np.mean(np.abs(tgt_slice - pred_slice))
        rmse = np.sqrt(np.mean((tgt_slice - pred_slice) ** 2))
        
        # Correlation (avoid NaN if std is 0)
        if np.std(tgt_slice) > 1e-8 and np.std(pred_slice) > 1e-8:
            corr = np.corrcoef(tgt_slice, pred_slice)[0, 1]
        else:
            corr = np.nan
        
        # Scaling factor: median(pred/tgt) where tgt > threshold
        threshold = np.max(tgt_slice) * 0.1
        mask = tgt_slice > threshold
        if np.sum(mask) > 10:
            ratio = np.median(pred_slice[mask] / (tgt_slice[mask] + 1e-8))
        else:
            ratio = np.nan
        
        # Max values in this layer
        tgt_max = np.max(tgt_slice)
        pred_max = np.max(pred_slice)
        
        stats_list.append({
            'layer': layer_idx,
            'tgt_max': tgt_max,
            'pred_max': pred_max,
            'mae': mae,
            'rmse': rmse,
            'corr': corr,
            'ratio': ratio,  # pred/tgt scaling
        })
    
    return stats_list

def plot_layer_analysis(stats_list, title="Layer-by-Layer Analysis"):
    """Create visualization of per-layer statistics"""
    layers = [s['layer'] for s in stats_list]
    mae = [s['mae'] for s in stats_list]
    rmse = [s['rmse'] for s in stats_list]
    corr = [s['corr'] for s in stats_list]
    ratio = [s['ratio'] for s in stats_list]
    tgt_max = [s['tgt_max'] for s in stats_list]
    pred_max = [s['pred_max'] for s in stats_list]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # MAE
    axes[0, 0].plot(layers, mae, 'o-', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Layer (z)')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Mean Absolute Error per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE
    axes[0, 1].plot(layers, rmse, 'o-', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Layer (z)')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Root Mean Squared Error per Layer')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation
    axes[0, 2].plot(layers, corr, 'o-', color='green', linewidth=2)
    axes[0, 2].set_xlabel('Layer (z)')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].set_title('Spatial Correlation per Layer')
    axes[0, 2].set_ylim([-0.1, 1.1])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Pred/Tgt Ratio
    axes[1, 0].plot(layers, ratio, 'o-', color='purple', linewidth=2)
    axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    axes[1, 0].set_xlabel('Layer (z)')
    axes[1, 0].set_ylabel('Pred/Tgt Ratio')
    axes[1, 0].set_title('Scaling Factor per Layer')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Max dose values
    axes[1, 1].plot(layers, tgt_max, 'o-', label='Target', linewidth=2, color='blue')
    axes[1, 1].plot(layers, pred_max, 's-', label='Prediction', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Layer (z)')
    axes[1, 1].set_ylabel('Max Dose Value')
    axes[1, 1].set_title('Maximum Dose per Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics summary text
    axes[1, 2].axis('off')
    summary_text = f"""
    Layer Statistics Summary:
    
    MAE:
      Mean: {np.nanmean(mae):.6f}
      Std:  {np.nanstd(mae):.6f}
      Max:  {np.nanmax(mae):.6f}
    
    RMSE:
      Mean: {np.nanmean(rmse):.6f}
      Std:  {np.nanstd(rmse):.6f}
    
    Correlation:
      Mean: {np.nanmean(corr):.4f}
      Min:  {np.nanmin(corr):.4f}
    
    Pred/Tgt Ratio:
      Mean: {np.nanmean(ratio):.4f}
      Std:  {np.nanstd(ratio):.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze model predictions layer-by-layer')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--model', type=str, default=None, help='Model path (auto-detect if not provided)')
    parser.add_argument('--layers', type=int, default=None, help='Number of layers to sample (default: all)')
    parser.add_argument('--output', type=str, default='layer_analysis.png', help='Output figure path')
    
    args = parser.parse_args()
    
    # Load model
    if args.model is None:
        model_path = get_best_model_path()
        print(f"Auto-detected model: {model_path}")
    else:
        model_path = Path(args.model)
    
    # Load architecture - define inline to avoid import issues
    import torch.nn as nn
    
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, base_channels=32):
            super().__init__()
            self.base_channels = base_channels
            
            # Encoder
            self.enc1 = self.conv_block(in_channels, base_channels)
            self.pool1 = nn.MaxPool3d(2)
            self.enc2 = self.conv_block(base_channels, base_channels*2)
            self.pool2 = nn.MaxPool3d(2)
            self.enc3 = self.conv_block(base_channels*2, base_channels*4)
            self.pool3 = nn.MaxPool3d(2)
            
            # Bottleneck
            self.bottleneck = self.conv_block(base_channels*4, base_channels*8)
            
            # Decoder
            self.upconv3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
            self.dec3 = self.conv_block(base_channels*8, base_channels*4)
            self.upconv2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(base_channels*4, base_channels*2)
            self.upconv1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(base_channels*2, base_channels)
            
            self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        def conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            enc3 = self.enc3(self.pool2(enc2))
            bottleneck = self.bottleneck(self.pool3(enc3))
            
            dec3 = self.dec3(torch.cat([self.upconv3(bottleneck), enc3], dim=1))
            dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
            dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
            
            return self.final(dec1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle checkpoint format - may be wrapped in 'model' key
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Load sample
    inp, tgt, sample_name = load_sample(args.sample)
    if inp is None:
        print(f"Sample {args.sample:03d} not found")
        return
    
    print(f"Loaded {sample_name}: Input {inp.shape}, Target {tgt.shape}")
    
    # Normalize
    inp_norm, tgt_norm = normalize_sample(inp, tgt)
    
    # Infer
    with torch.no_grad():
        inp_tensor = torch.from_numpy(inp_norm[np.newaxis, np.newaxis]).float().to(device)
        pred_tensor = model(inp_tensor)
        pred_norm = pred_tensor.cpu().numpy()[0, 0]
    
    print(f"Prediction shape: {pred_norm.shape}")
    print(f"Target range: [{tgt_norm.min():.6f}, {tgt_norm.max():.6f}]")
    print(f"Prediction range: [{pred_norm.min():.6f}, {pred_norm.max():.6f}]")
    print()
    
    # Analyze layers
    stats_list = analyze_layers(tgt_norm, pred_norm, num_layers=args.layers)
    
    # Print detailed table
    print(f"{'Layer':<6} {'Tgt Max':<10} {'Pred Max':<10} {'MAE':<10} {'RMSE':<10} {'Corr':<8} {'Ratio':<8}")
    print("-" * 72)
    for s in stats_list:
        print(f"{s['layer']:<6d} {s['tgt_max']:<10.6f} {s['pred_max']:<10.6f} "
              f"{s['mae']:<10.6f} {s['rmse']:<10.6f} {s['corr']:<8.4f} {s['ratio']:<8.4f}")
    
    # Plot
    fig = plot_layer_analysis(stats_list, title=f"Layer-by-Layer Analysis - {sample_name}")
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {args.output}")
    plt.show()

if __name__ == '__main__':
    main()
