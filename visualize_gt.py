#!/usr/bin/env python3
"""
Visualize ground truth volume with selected slices and statistics.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import argparse

def load_gt(filepath):
    gt = np.load(filepath)
    return gt

def plot_gt_slices_and_profiles(gt, output_path='gt_visualization.png'):
    """
    Visualize GT: slices + dose profiles + statistics
    """
    D, H, W = gt.shape
    
    # Select slices: entrance, mid-range (peak), and exit
    slices_to_show = [
        (0, "Entrance (z=0)"),
        (D//4, f"Buildup (z={D//4})"),
        (D//2, f"Peak (z={D//2})"),
        (3*D//4, f"Falloff (z={3*D//4})"),
        (D-1, f"Exit (z={D-1})")
    ]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.35)
    
    fig.suptitle("Ground Truth Volume Visualization (29.4M particles)", fontsize=16, fontweight='bold')
    
    # Row 1: Dose maps for selected slices
    ax_slices = []
    for i, (z_idx, label) in enumerate(slices_to_show):
        if i < 5:
            ax = fig.add_subplot(gs[0, i])
            slice_2d = gt[z_idx]
            
            # Use log scale for visualization
            im = ax.imshow(slice_2d, cmap='hot', norm=LogNorm(vmin=np.max(slice_2d)*0.001, vmax=np.max(slice_2d)))
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Dose (log scale)')
            ax_slices.append((ax, z_idx, slice_2d))
    
    # Row 2: Depth dose profiles (PDD)
    ax_pdd = fig.add_subplot(gs[1, :3])
    
    # Calculate PDD - dose along z-axis averaged over central region
    h_center = H // 2
    w_center = W // 2
    region_size = 20
    
    central_dose = gt[
        :,
        h_center - region_size:h_center + region_size,
        w_center - region_size:w_center + region_size
    ].mean(axis=(1, 2))
    
    ax_pdd.plot(range(D), central_dose, 'b-', linewidth=2, label='Central axis PDD')
    ax_pdd.fill_between(range(D), 0, central_dose, alpha=0.3)
    ax_pdd.set_xlabel('Depth (z layer)')
    ax_pdd.set_ylabel('Dose (arbitrary units)')
    ax_pdd.set_title('Percent Depth Dose (PDD) - Central Axis', fontweight='bold')
    ax_pdd.grid(True, alpha=0.3)
    ax_pdd.legend()
    
    # Row 2: Lateral dose profiles
    ax_lateral_x = fig.add_subplot(gs[1, 3:])
    
    # Lateral profile at peak dose depth
    peak_z = np.argmax(central_dose)
    profile_x = gt[peak_z, h_center, :]
    profile_y = gt[peak_z, :, w_center]
    
    ax_lateral_x.plot(profile_x, 'r-', linewidth=2, label='X-axis profile')
    ax_lateral_x.plot(profile_y, 'b-', linewidth=2, label='Y-axis profile')
    ax_lateral_x.set_xlabel('Distance (voxels)')
    ax_lateral_x.set_ylabel('Dose')
    ax_lateral_x.set_title(f'Lateral Profiles at Peak Depth (z={peak_z})', fontweight='bold')
    ax_lateral_x.grid(True, alpha=0.3)
    ax_lateral_x.legend()
    
    # Row 3: Statistics and metrics
    ax_stats = fig.add_subplot(gs[2, :3])
    ax_stats.axis('off')
    
    # Calculate statistics
    nonzero = gt[gt > 0]
    voxels_above_50pct = np.sum(gt > np.max(gt) * 0.5)
    voxels_above_80pct = np.sum(gt > np.max(gt) * 0.8)
    voxels_above_90pct = np.sum(gt > np.max(gt) * 0.9)
    
    stats_text = f"""
    GROUND TRUTH STATISTICS (29.4M particles)
    
    Volume: {D} × {H} × {W} = {D*H*W:,} voxels
    
    Dose Range:
      Min: {np.min(gt):.6f}
      Max: {np.max(gt):.6f}
      Mean: {np.mean(nonzero):.6f} (non-zero only)
    
    Dose Levels:
      > 50% of max: {voxels_above_50pct:,} voxels
      > 80% of max: {voxels_above_80pct:,} voxels
      > 90% of max: {voxels_above_90pct:,} voxels
    
    PDD Info:
      Peak depth: z = {peak_z}
      Peak dose: {central_dose[peak_z]:.6f}
      Entrance dose: {central_dose[0]:.6f}
      Dose at exit: {central_dose[-1]:.6f}
    
    Symmetry:
      X-profile asymmetry: {np.abs(np.max(profile_x) - np.max(np.flip(profile_x))) / np.max(profile_x) * 100:.2f}%
      Y-profile asymmetry: {np.abs(np.max(profile_y) - np.max(np.flip(profile_y))) / np.max(profile_y) * 100:.2f}%
    """
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 3: Dose histogram
    ax_hist = fig.add_subplot(gs[2, 3:])
    
    nonzero_doses = gt[gt > 0]
    ax_hist.hist(nonzero_doses, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.axvline(np.mean(nonzero_doses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(nonzero_doses):.2f}')
    ax_hist.axvline(np.median(nonzero_doses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(nonzero_doses):.2f}')
    ax_hist.set_xlabel('Dose Value')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Dose Histogram (non-zero voxels)', fontweight='bold')
    ax_hist.set_yscale('log')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3, which='both')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize ground truth')
    parser.add_argument('gtfile', type=str, help='Ground truth file')
    parser.add_argument('--output', type=str, default='gt_visualization.png', help='Output path')
    
    args = parser.parse_args()
    
    gtfile = Path(args.gtfile)
    if not gtfile.exists():
        print(f"File not found: {gtfile}")
        return
    
    gt = load_gt(gtfile)
    print(f"Loaded GT: {gt.shape}")
    plot_gt_slices_and_profiles(gt, args.output)

if __name__ == '__main__':
    main()
