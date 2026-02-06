#!/usr/bin/env python3
"""
Analyze ground truth quality layer-by-layer across the volume.
Shows smoothness, symmetry, SNR, and consistency for each z-slice.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
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
                'cv': np.nan,  # Coefficient of variation
            })
            continue
        
        flat = slice_2d.flatten()
        
        # Basic statistics
        max_dose = np.max(slice_2d)
        mean_dose = np.mean(flat)
        std_dose = np.std(flat)
        cv = std_dose / (mean_dose + 1e-8)  # Coefficient of variation
        
        # SNR: ratio of signal to noise
        # Signal = mean of nonzero voxels, Noise = std of periphery
        beam_region = slice_2d > np.percentile(flat, 50)
        if np.sum(beam_region) > 10:
            signal = np.mean(slice_2d[beam_region])
            noise = np.std(slice_2d[~beam_region])
            snr = signal / (noise + 1e-8)
        else:
            snr = np.nan
        
        # Smoothness: compute second derivative
        # Smooth slices have low second derivatives
        if slice_2d.shape[0] > 3 and slice_2d.shape[1] > 3:
            # Gaussian filtering to emphasize smooth vs noisy
            smooth = gaussian_filter(slice_2d, sigma=1.0)
            roughness = np.mean(np.abs(slice_2d - smooth))
            smoothness = 1.0 / (1.0 + roughness)  # Higher = smoother
        else:
            smoothness = np.nan
        
        # Symmetry (X and Y axis)
        h, w = slice_2d.shape
        mid_h, mid_w = h // 2, w // 2
        
        # X-axis symmetry (left-right)
        left = slice_2d[:, :mid_w]
        right = slice_2d[:, mid_w:]
        if right.shape[1] < left.shape[1]:
            left = left[:, :right.shape[1]]
        if right.shape[1] > 0 and left.shape[1] > 0:
            asym_x = np.mean(np.abs(left - np.fliplr(right))) / (np.mean(np.abs(slice_2d)) + 1e-8)
            asym_x = asym_x * 100  # as percentage
        else:
            asym_x = np.nan
        
        # Y-axis symmetry (up-down)
        top = slice_2d[:mid_h, :]
        bottom = slice_2d[mid_h:, :]
        if bottom.shape[0] < top.shape[0]:
            top = top[:bottom.shape[0], :]
        if bottom.shape[0] > 0 and top.shape[0] > 0:
            asym_y = np.mean(np.abs(top - np.flipud(bottom))) / (np.mean(np.abs(slice_2d)) + 1e-8)
            asym_y = asym_y * 100  # as percentage
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

def plot_gt_quality(stats_list, gt_filepath, title="Ground Truth Quality Analysis"):
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
    
    # Max Dose
    axes[0, 0].plot(layers, max_dose, 'o-', color='red', linewidth=2)
    axes[0, 0].set_xlabel('Layer (z)')
    axes[0, 0].set_ylabel('Max Dose')
    axes[0, 0].set_title('Maximum Dose per Layer')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean Dose
    axes[0, 1].plot(layers, mean_dose, 'o-', color='blue', linewidth=2)
    axes[0, 1].set_xlabel('Layer (z)')
    axes[0, 1].set_ylabel('Mean Dose')
    axes[0, 1].set_title('Mean Dose per Layer')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coefficient of Variation (lower = less noisy)
    axes[0, 2].plot(layers, cv, 'o-', color='green', linewidth=2)
    axes[0, 2].set_xlabel('Layer (z)')
    axes[0, 2].set_ylabel('Coeff. of Variation')
    axes[0, 2].set_title('Dose Variation (CV)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # SNR (higher = cleaner)
    axes[0, 3].plot(layers, snr, 'o-', color='purple', linewidth=2)
    axes[0, 3].set_xlabel('Layer (z)')
    axes[0, 3].set_ylabel('SNR')
    axes[0, 3].set_title('Signal-to-Noise Ratio')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Smoothness (higher = smoother)
    axes[1, 0].plot(layers, smoothness, 'o-', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Layer (z)')
    axes[1, 0].set_ylabel('Smoothness')
    axes[1, 0].set_title('Dose Smoothness (1=perfect)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # X-axis Asymmetry (lower = more symmetric)
    axes[1, 1].plot(layers, asym_x, 'o-', color='brown', linewidth=2)
    axes[1, 1].set_xlabel('Layer (z)')
    axes[1, 1].set_ylabel('Asymmetry (%)')
    axes[1, 1].set_title('X-axis Asymmetry')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Y-axis Asymmetry (lower = more symmetric)
    axes[1, 2].plot(layers, asym_y, 'o-', color='teal', linewidth=2)
    axes[1, 2].set_xlabel('Layer (z)')
    axes[1, 2].set_ylabel('Asymmetry (%)')
    axes[1, 2].set_title('Y-axis Asymmetry')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Statistics summary
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
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze ground truth quality layer-by-layer')
    parser.add_argument('gtfile', type=str, help='Ground truth .npy file path')
    parser.add_argument('--layers', type=int, default=None, help='Number of layers to sample (default: all)')
    parser.add_argument('--output', type=str, default='gt_quality_layers.png', help='Output figure path')
    
    args = parser.parse_args()
    
    # Load GT
    gtfile = Path(args.gtfile)
    if not gtfile.exists():
        print(f"File not found: {gtfile}")
        return
    
    gt = load_ground_truth(gtfile)
    print(f"Loaded GT: {gt.shape}")
    print(f"Range: [{gt.min():.6f}, {gt.max():.6f}]")
    print(f"Mean: {gt.mean():.6f}, Std: {gt.std():.6f}")
    print()
    
    # Analyze layers
    stats_list = analyze_gt_layers(gt, num_layers=args.layers)
    
    # Filter out empty layers
    stats_valid = [s for s in stats_list if not s['is_empty']]
    
    # Print detailed table
    print(f"{'Layer':<6} {'Max':<10} {'Mean':<10} {'CV':<8} {'SNR':<8} {'Smooth':<8} {'AsymX%':<8} {'AsymY%':<8}")
    print("-" * 86)
    for s in stats_valid:
        print(f"{s['layer']:<6d} {s['max_dose']:<10.6f} {s['mean_dose']:<10.6f} "
              f"{s['cv']:<8.4f} {s['snr']:<8.4f} {s['smoothness']:<8.4f} "
              f"{s['symmetry_x']:<8.2f} {s['symmetry_y']:<8.2f}")
    
    # Overall statistics
    print("\n" + "="*86)
    print("OVERALL GROUND TRUTH QUALITY")
    print("="*86)
    max_doses = [s['max_dose'] for s in stats_valid]
    cvs = [s['cv'] for s in stats_valid]
    snrs = [s['snr'] for s in stats_valid]
    smooths = [s['smoothness'] for s in stats_valid]
    asym_xs = [s['symmetry_x'] for s in stats_valid]
    asym_ys = [s['symmetry_y'] for s in stats_valid]
    
    print(f"Max Dose:           {np.mean(max_doses):.6f} ± {np.std(max_doses):.6f}")
    print(f"CV (variation):     {np.mean(cvs):.4f} ± {np.std(cvs):.4f} (lower=better)")
    print(f"SNR:                {np.mean(snrs):.4f} ± {np.std(snrs):.4f} (higher=better)")
    print(f"Smoothness:         {np.mean(smooths):.4f} ± {np.std(smooths):.4f} (higher=smoother)")
    print(f"Asymmetry X:        {np.mean(asym_xs):.2f}% ± {np.std(asym_xs):.2f}% (lower=more symmetric)")
    print(f"Asymmetry Y:        {np.mean(asym_ys):.2f}% ± {np.std(asym_ys):.2f}%")
    
    # Plot
    fig = plot_gt_quality(stats_valid, str(gtfile), title="Ground Truth (29.4M) Layer Quality")
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {args.output}")

if __name__ == '__main__':
    main()
