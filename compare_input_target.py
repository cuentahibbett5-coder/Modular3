#!/usr/bin/env python3
"""
Compare input and ground truth layer-by-layer to understand scaling behavior.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def load_sample(idx, data_dir='data'):
    """Load input and target from dataset"""
    inp_path = Path(data_dir) / f'input_{idx:03d}.npy'
    tgt_path = Path(data_dir) / f'target_{idx:03d}.npy'
    
    if not inp_path.exists() or not tgt_path.exists():
        return None, None
    
    inp = np.load(inp_path)
    tgt = np.load(tgt_path)
    return inp, tgt

def analyze_layer_profiles(inp, tgt, num_samples=20):
    """
    Compare input and target layer profiles.
    Returns statistics for understanding the amplification factor.
    """
    D = inp.shape[0]
    layers = np.linspace(0, D-1, num_samples, dtype=int)
    
    stats_list = []
    
    for layer_idx in layers:
        inp_slice = inp[layer_idx].flatten()
        tgt_slice = tgt[layer_idx].flatten()
        
        # Skip if both are mostly zero
        if np.sum(inp_slice) < 1e-6 or np.sum(tgt_slice) < 1e-6:
            continue
        
        inp_max = np.max(inp_slice)
        tgt_max = np.max(tgt_slice)
        
        # Amplification factor: target / input
        # Use median of high-dose voxels to avoid outliers
        threshold_inp = np.max(inp_slice) * 0.1
        threshold_tgt = np.max(tgt_slice) * 0.1
        
        mask_inp = inp_slice > threshold_inp
        mask_tgt = tgt_slice > threshold_tgt
        mask_both = mask_inp & mask_tgt
        
        if np.sum(mask_both) > 10:
            ratio = np.median(tgt_slice[mask_both] / (inp_slice[mask_both] + 1e-8))
        else:
            ratio = np.nan
        
        stats_list.append({
            'layer': layer_idx,
            'inp_max': inp_max,
            'tgt_max': tgt_max,
            'amplification': tgt_max / (inp_max + 1e-8),
            'ratio_median': ratio,
        })
    
    return stats_list

def plot_comparison(stats_list, sample_id, output_path='input_target_comparison.png'):
    """Visualize input vs target relationship"""
    layers = [s['layer'] for s in stats_list]
    inp_max = [s['inp_max'] for s in stats_list]
    tgt_max = [s['tgt_max'] for s in stats_list]
    amplif = [s['amplification'] for s in stats_list]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Input vs Target Analysis - Sample {sample_id:03d}", fontsize=14, fontweight='bold')
    
    # Max values comparison
    axes[0].plot(layers, inp_max, 'o-', label='Input Max', linewidth=2, color='blue')
    axes[0].plot(layers, tgt_max, 's-', label='Target Max', linewidth=2, color='red')
    axes[0].set_xlabel('Layer (z)')
    axes[0].set_ylabel('Max Dose Value')
    axes[0].set_title('Maximum Dose per Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale to see the relationship better
    axes[1].loglog(inp_max, tgt_max, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].loglog([min(inp_max), max(tgt_max)], 
                    [min(inp_max), max(tgt_max)], 'k--', alpha=0.5, label='y=x (no amplification)')
    axes[1].set_xlabel('Input Max (log)')
    axes[1].set_ylabel('Target Max (log)')
    axes[1].set_title('Input vs Target (log-log)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    # Amplification factor
    axes[2].plot(layers, amplif, 'o-', linewidth=2, color='purple')
    axes[2].axhline(y=np.nanmean(amplif), color='r', linestyle='--', label=f'Mean: {np.nanmean(amplif):.2f}')
    axes[2].set_xlabel('Layer (z)')
    axes[2].set_ylabel('Target/Input Ratio')
    axes[2].set_title('Amplification Factor per Layer')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare input and target layer-by-layer')
    parser.add_argument('--sample', type=int, default=0, help='Sample index')
    parser.add_argument('--output', type=str, default='input_target_comparison.png', help='Output path')
    args = parser.parse_args()
    
    inp, tgt = load_sample(args.sample)
    if inp is None:
        print(f"Sample {args.sample:03d} not found")
        return
    
    print(f"Sample {args.sample:03d}:")
    print(f"  Input shape: {inp.shape}, range: [{inp.min():.6f}, {inp.max():.6f}]")
    print(f"  Target shape: {tgt.shape}, range: [{tgt.min():.6f}, {tgt.max():.6f}]")
    print()
    
    stats_list = analyze_layer_profiles(inp, tgt, num_samples=30)
    
    print(f"{'Layer':<6} {'Input Max':<12} {'Target Max':<12} {'Amplif. (Tgt/Inp)':<18}")
    print("-" * 60)
    for s in stats_list:
        print(f"{s['layer']:<6d} {s['inp_max']:<12.6f} {s['tgt_max']:<12.6f} {s['amplification']:<18.6f}")
    
    print()
    print("="*60)
    print("STATISTICS")
    print("="*60)
    amplifs = [s['amplification'] for s in stats_list if not np.isnan(s['amplification'])]
    print(f"Mean Amplification: {np.mean(amplifs):.6f}")
    print(f"Std Amplification:  {np.std(amplifs):.6f}")
    print(f"Min Amplification:  {np.min(amplifs):.6f}")
    print(f"Max Amplification:  {np.max(amplifs):.6f}")
    print()
    print("Interpretation:")
    print("- If amplification ≈ 1: Target ≈ Input (no amplification)")
    print("- If amplification >> 1: Target >> Input (strong amplification)")
    print("- If constant: Scaling is linear")
    print("- If variable: Scaling depends on dose level")
    
    plot_comparison(stats_list, args.sample, args.output)

if __name__ == '__main__':
    main()
