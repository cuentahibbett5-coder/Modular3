#!/usr/bin/env python3
"""
Análisis rápido de predicciones exportadas
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EXPORTS_DIR = Path("exports")

def analyze_prediction(pair_name, level):
    """Analiza una predicción específica"""
    prefix = f"{pair_name}_{level}"
    
    input_vol = np.load(EXPORTS_DIR / f"{prefix}_input.npy")
    pred_vol = np.load(EXPORTS_DIR / f"{prefix}_pred.npy")
    target_vol = np.load(EXPORTS_DIR / f"{prefix}_target.npy")
    
    print(f"\n{'='*70}")
    print(f"{pair_name} - {level}")
    print(f"{'='*70}")
    print(f"Shape: {pred_vol.shape}")
    print(f"\nDose Statistics:")
    print(f"  Target max:  {target_vol.max():.1f}")
    print(f"  Input max:   {input_vol.max():.1f}")
    print(f"  Pred max:    {pred_vol.max():.1f}")
    print(f"\nMean Absolute Error:")
    print(f"  Input vs GT: {np.abs(input_vol - target_vol).mean():.2f}")
    print(f"  Pred vs GT:  {np.abs(pred_vol - target_vol).mean():.2f}")
    
    # PDD Comparison
    z_size = target_vol.shape[0]
    pdd_input = np.array([input_vol[z].max() for z in range(z_size)])
    pdd_pred = np.array([pred_vol[z].max() for z in range(z_size)])
    pdd_gt = np.array([target_vol[z].max() for z in range(z_size)])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pdd_gt, 'k-', linewidth=2, label='GT', alpha=0.8)
    plt.plot(pdd_input, 'r--', linewidth=1, label='Input', alpha=0.6)
    plt.plot(pdd_pred, 'b-', linewidth=1.5, label='Pred', alpha=0.8)
    plt.xlabel('Z layer')
    plt.ylabel('Max Dose')
    plt.title(f'PDD Curves - {pair_name} {level}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    err_input = np.abs(pdd_input - pdd_gt) / pdd_gt * 100
    err_pred = np.abs(pdd_pred - pdd_gt) / pdd_gt * 100
    plt.plot(err_input, 'r--', linewidth=1, label='Input error %', alpha=0.6)
    plt.plot(err_pred, 'b-', linewidth=1.5, label='Pred error %', alpha=0.8)
    plt.xlabel('Z layer')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error in PDD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(EXPORTS_DIR / f"analysis_{prefix}.png", dpi=150)
    print(f"\n✓ Plot saved: analysis_{prefix}.png")
    plt.close()
    
    # Central slice comparison
    z_mid = z_size // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    vmax = target_vol.max()
    
    im0 = axes[0, 0].imshow(input_vol[z_mid], cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title(f'Input {level}')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(pred_vol[z_mid], cmap='hot', vmin=0, vmax=vmax)
    axes[0, 1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(target_vol[z_mid], cmap='hot', vmin=0, vmax=vmax)
    axes[0, 2].set_title('Ground Truth')
    plt.colorbar(im2, ax=axes[0, 2])
    
    diff_input = np.abs(input_vol[z_mid] - target_vol[z_mid])
    diff_pred = np.abs(pred_vol[z_mid] - target_vol[z_mid])
    
    im3 = axes[1, 0].imshow(diff_input, cmap='viridis')
    axes[1, 0].set_title('|Input - GT|')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(diff_pred, cmap='viridis')
    axes[1, 1].set_title('|Pred - GT|')
    plt.colorbar(im4, ax=axes[1, 1])
    
    mask = target_vol[z_mid] > 0.01 * vmax
    rel_err = np.zeros_like(target_vol[z_mid])
    if mask.any():
        rel_err[mask] = np.abs(pred_vol[z_mid][mask] - target_vol[z_mid][mask]) / target_vol[z_mid][mask] * 100
    
    im5 = axes[1, 2].imshow(rel_err, cmap='RdYlGn_r', vmin=0, vmax=50)
    axes[1, 2].set_title('Relative Error %')
    plt.colorbar(im5, ax=axes[1, 2])
    
    for ax in axes.flat:
        ax.axis('off')
    
    fig.suptitle(f'{pair_name} {level} - Central Slice (z={z_mid})', fontsize=14)
    plt.tight_layout()
    plt.savefig(EXPORTS_DIR / f"slice_{prefix}.png", dpi=150)
    print(f"✓ Plot saved: slice_{prefix}.png")
    plt.close()

def main():
    print("="*70)
    print("ANÁLISIS DE PREDICCIONES EXPORTADAS")
    print("="*70)
    
    # Find all exported predictions
    pred_files = sorted(EXPORTS_DIR.glob("*_pred.npy"))
    
    if not pred_files:
        print("\n⚠ No prediction files found in exports/")
        return
    
    print(f"\nFound {len(pred_files)} predictions")
    
    for pred_file in pred_files:
        parts = pred_file.stem.replace("_pred", "").split("_")
        pair_name = "_".join(parts[:-1])
        level = parts[-1]
        
        try:
            analyze_prediction(pair_name, level)
        except Exception as e:
            print(f"\n⚠ Error analyzing {pair_name} {level}: {e}")
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETO")
    print("="*70)

if __name__ == "__main__":
    main()
