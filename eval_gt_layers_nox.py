#!/usr/bin/env python3
"""
Analyze ground truth quality layer-by-layer across the volume (non-interactive).
Shows smoothness, symmetry, SNR, and consistency for each z-slice.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse

def load_ground_truth(filepath):
    """Load ground truth dose distribution"""
    gt = np.load(filepath)
    return gt

def analyze_gt_layers(gt, num_layers=None):
    """
    Analyze ground truth quality in each layer
    
    Args:
        gt: (D, H, W) ground truth volume
        num_layers: number of layers to analyze (default: all)
    
    Returns:
        list of dicts with per-layer statistics
    """
    D = gt.shape[0]
    if num_layers is None:
        num_layers = D
    
    layers = np.linspace(0, D-1, num_layers, dtype=int)
    
    stats_list = []
    
    for layer_idx in layers:
        slice_2d = gt[layer_idx]
        
        # Skip if slice is mostly zeros
        if np.sum(slice_2d) < 1e-6:
            stats_list.append({
                'layer': layer_idx,
                'is_empty': True,
                'max_dose': 0,
                'mean_dose': 0,
                'std_dose': 0,
                'snr': np.nan,
                'smoothness': np.nan,
                'symmetry_x': np.nan,
                'symmetry_y': np.nan,
                'cv': np.nan,
            })
            continue
        
        flat = slice_2d.flatten()
        
        # Basic statistics
        max_dose = np.max(slice_2d)
        mean_dose = np.mean(flat)
        std_dose = np.std(flat)
        cv = std_dose / (mean_dose + 1e-8)
        
        # SNR
        beam_region = slice_2d > np.percentile(flat, 50)
        if np.sum(beam_region) > 10:
            signal = np.mean(slice_2d[beam_region])
            noise = np.std(slice_2d[~beam_region])
            snr = signal / (noise + 1e-8)
        else:
            snr = np.nan
        
        # Smoothness
        if slice_2d.shape[0] > 3 and slice_2d.shape[1] > 3:
            smooth = gaussian_filter(slice_2d, sigma=1.0)
            roughness = np.mean(np.abs(slice_2d - smooth))
            smoothness = 1.0 / (1.0 + roughness)
        else:
            smoothness = np.nan
        
        # Symmetry (X and Y axis)
        h, w = slice_2d.shape
        mid_h, mid_w = h // 2, w // 2
        
        # X-axis symmetry
        left = slice_2d[:, :mid_w]
        right = slice_2d[:, mid_w:]
        if right.shape[1] < left.shape[1]:
            left = left[:, :right.shape[1]]
        if right.shape[1] > 0 and left.shape[1] > 0:
            asym_x = np.mean(np.abs(left - np.fliplr(right))) / (np.mean(np.abs(slice_2d)) + 1e-8)
            asym_x = asym_x * 100
        else:
            asym_x = np.nan
        
        # Y-axis symmetry
        top = slice_2d[:mid_h, :]
        bottom = slice_2d[mid_h:, :]
        if bottom.shape[0] < top.shape[0]:
            top = top[:bottom.shape[0], :]
        if bottom.shape[0] > 0 and top.shape[0] > 0:
            asym_y = np.mean(np.abs(top - np.flipud(bottom))) / (np.mean(np.abs(slice_2d)) + 1e-8)
            asym_y = asym_y * 100
        else:
            asym_y = np.nan
        
        stats_list.append({
            'layer': layer_idx,
            'is_empty': False,
            'max_dose': max_dose,
            'mean_dose': mean_dose,
            'std_dose': std_dose,
            'cv': cv,
            'snr': snr,
            'smoothness': smoothness,
            'symmetry_x': asym_x,
            'symmetry_y': asym_y,
        })
    
    return stats_list

def plot_gt_quality(stats_list, gt_filepath, title="Ground Truth Quality Analysis", output_path='gt_quality.png'):
    """Create visualization of GT quality per layer"""
    layers = [s['layer'] for s in stats_list if not s['is_empty']]
    max_dose = [s['max_dose'] for s in stats_list if not s['is_empty']]
    mean_dose = [s['mean_dose'] for s in stats_list if not s['is_empty']]
    cv = [s['cv'] for s in stats_list if not s['is_empty']]
    snr = [s['snr'] for s in stats_list if not s['is_empty']]
    smoothness = [s['smoothness'] for s in stats_list if not s['is_empty']]
    asym_x = [s['symmetry_x'] for s in stats_list if not s['is_empty']]
    asym_y = [s['symmetry_y'] for s in stats_list if not s['is_empty']]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(f"{title}\n{gt_filepath}", fontsize=14, fontweight='bold')
    
    axes[0, 0].plot(layers, max_dose, 'o-', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Layer (z)')
    axes[0, 0].set_ylabel('Max Dose')
    axes[0, 0].set_title('Maximum Dose per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(layers, mean_dose, 'o-', color='blue', linewidth=2)
    axes[0, 1].set_xlabel('Layer (z)')
    axes[0, 1].set_ylabel('Mean Dose')
    axes[0, 1].set_title('Mean Dose per Layer')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(layers, cv, 'o-', color='green', linewidth=2)
    axes[0, 2].set_xlabel('Layer (z)')
    axes[0, 2].set_ylabel('Coeff. of Variation')
    axes[0, 2].set_title('Dose Variation (CV)')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].plot(layers, snr, 'o-', color='purple', linewidth=2)
    axes[0, 3].set_xlabel('Layer (z)')
    axes[0, 3].set_ylabel('SNR')
    axes[0, 3].set_title('Signal-to-Noise Ratio')
    axes[0, 3].grid(True, alpha=0.3)
    
    axes[1, 0].plot(layers, smoothness, 'o-', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Layer (z)')
    axes[1, 0].set_ylabel('Smoothness')
    axes[1, 0].set_title('Dose Smoothness (1=perfect)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(layers, asym_x, 'o-', color='brown', linewidth=2)
    axes[1, 1].set_xlabel('Layer (z)')
    axes[1, 1].set_ylabel('Asymmetry (%)')
    axes[1, 1].set_title('X-axis Asymmetry')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(layers, asym_y, 'o-', color='teal', linewidth=2)
    axes[1, 2].set_xlabel('Layer (z)')
    axes[1, 2].set_ylabel('Asymmetry (%)')
    axes[1, 2].set_title('Y-axis Asymmetry')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].axis('off')
    summary_text = f"""
    Overall GT Quality Metrics:
    
    Max Dose:
      Mean: {np.nanmean(max_dose):.6f}
      Std:  {np.nanstd(max_dose):.6f}
    
    CV (Variation):
      Mean: {np.nanmean(cv):.4f}
      Max:  {np.nanmax(cv):.4f}
    
    SNR (Signal/Noise):
      Mean: {np.nanmean(snr):.4f}
      Min:  {np.nanmin(snr):.4f}
    
    Smoothness:
      Mean: {np.nanmean(smoothness):.4f}
      Min:  {np.nanmin(smoothness):.4f}
    
    Asymmetry X/Y:
      Mean X: {np.nanmean(asym_x):.2f}%
      Mean Y: {np.nanmean(asym_y):.2f}%
    """
    axes[1, 3].text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 3].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ground truth quality layer-by-layer')
    parser.add_argument('gtfile', type=str, help='Ground truth .npy file path')
    parser.add_argument('--layers', type=int, default=None, help='Number of layers to sample (default: all)')
    parser.add_argument('--output', type=str, default='gt_quality_layers.png', help='Output figure path')
    
    args = parser.parse_args()
    
    gtfile = Path(args.gtfile)
    if not gtfile.exists():
        print(f"File not found: {gtfile}")
        return
    
    gt = load_ground_truth(gtfile)
    print(f"Loaded GT: {gt.shape}")
    print(f"Range: [{gt.min():.6f}, {gt.max():.6f}]")
    print(f"Mean: {gt.mean():.6f}, Std: {gt.std():.6f}\n")
    
    stats_list = analyze_gt_layers(gt, num_layers=args.layers)
    stats_valid = [s for s in stats_list if not s['is_empty']]
    
    # Summary
    max_doses = [s['max_dose'] for s in stats_valid]
    cvs = [s['cv'] for s in stats_valid]
    snrs = [s['snr'] for s in stats_valid]
    smooths = [s['smoothness'] for s in stats_valid]
    asym_xs = [s['symmetry_x'] for s in stats_valid]
    asym_ys = [s['symmetry_y'] for s in stats_valid]
    
    print("="*86)
    print("GROUND TRUTH QUALITY SUMMARY (all layers)")
    print("="*86)
    print(f"Max Dose:           {np.mean(max_doses):.6f} ± {np.std(max_doses):.6f}")
    print(f"CV (variation):     {np.mean(cvs):.4f} ± {np.std(cvs):.4f} (lower=better)")
    print(f"SNR:                {np.mean(snrs):.4f} ± {np.std(snrs):.4f} (higher=better)")
    print(f"Smoothness:         {np.mean(smooths):.4f} ± {np.std(smooths):.4f} (higher=smoother)")
    print(f"Asymmetry X:        {np.mean(asym_xs):.2f}% ± {np.std(asym_xs):.2f}% (lower=better)")
    print(f"Asymmetry Y:        {np.mean(asym_ys):.2f}% ± {np.std(asym_ys):.2f}%")
    print()
    
    # Analysis by region
    print("="*86)
    print("ANALYSIS BY REGION")
    print("="*86)
    
    core_range = 50  # Central 50 layers
    core_start = max(0, 150 - core_range//2)
    core_end = min(300, 150 + core_range//2)
    core_stats = [s for s in stats_valid if core_start <= s['layer'] < core_end]
    
    periph_stats = [s for s in stats_valid if s['layer'] < core_start or s['layer'] >= core_end]
    
    print(f"\nCORE (layers {core_start}-{core_end}):")
    print(f"  Max Dose:      {np.mean([s['max_dose'] for s in core_stats]):.6f}")
    print(f"  CV:            {np.mean([s['cv'] for s in core_stats]):.4f}")
    print(f"  SNR:           {np.mean([s['snr'] for s in core_stats]):.4f}")
    print(f"  Smoothness:    {np.mean([s['smoothness'] for s in core_stats]):.4f}")
    print(f"  Asymmetry:     {np.mean([s['symmetry_x'] for s in core_stats]):.2f}%")
    
    print(f"\nPERIPHERY (layers 0-{core_start} and {core_end}-299):")
    print(f"  Max Dose:      {np.mean([s['max_dose'] for s in periph_stats]):.6f}")
    print(f"  CV:            {np.mean([s['cv'] for s in periph_stats]):.4f}")
    print(f"  SNR:           {np.mean([s['snr'] for s in periph_stats]):.4f}")
    print(f"  Smoothness:    {np.mean([s['smoothness'] for s in periph_stats]):.4f}")
    print(f"  Asymmetry:     {np.mean([s['symmetry_x'] for s in periph_stats]):.2f}%")
    
    # Plot
    plot_gt_quality(stats_valid, str(gtfile), title="Ground Truth (29.4M) Layer Quality", output_path=args.output)

if __name__ == '__main__':
    main()
