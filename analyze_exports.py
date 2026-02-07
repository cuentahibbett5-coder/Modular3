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
    
    # Central slice - SOLO PREDICCIÓN
    z_mid = z_size // 2
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    vmax = pred_vol.max()
    
    im = ax.imshow(pred_vol[z_mid], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
    ax.set_title(f'Prediction - {pair_name} {level} (z={z_mid})', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(EXPORTS_DIR / f"pred_{prefix}_z{z_mid}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: pred_{prefix}_z{z_mid}.png")
    plt.close()
    
    # Cortes en z = 0, 5, 10, 15 - COMPARACIÓN COMPLETA
    z_levels = [0, 5, 10, 15]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    vmax = target_vol.max()
    
    for idx, z in enumerate(z_levels):
        # Input
        im0 = axes[idx, 0].imshow(input_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[idx, 0].set_ylabel(f'z = {z}', fontsize=12, fontweight='bold')
        if idx == 0:
            axes[idx, 0].set_title('Input', fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046)
        
        # Prediction
        im1 = axes[idx, 1].imshow(pred_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        if idx == 0:
            axes[idx, 1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
        
        # Target
        im2 = axes[idx, 2].imshow(target_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        if idx == 0:
            axes[idx, 2].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
        
        # Error relativo
        mask = target_vol[z] > 0.01 * vmax
        rel_err = np.zeros_like(target_vol[z])
        if mask.any():
            rel_err[mask] = np.abs(pred_vol[z][mask] - target_vol[z][mask]) / target_vol[z][mask] * 100
        
        im3 = axes[idx, 3].imshow(rel_err, cmap='RdYlGn_r', vmin=0, vmax=50, aspect='auto')
        if idx == 0:
            axes[idx, 3].set_title('Error % (Pred vs GT)', fontsize=14, fontweight='bold')
        axes[idx, 3].axis('off')
        plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046, label='%')
    
    fig.suptitle(f'{pair_name} {level} - Comparison at z=0,5,10,15', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(EXPORTS_DIR / f"comparison_{prefix}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: comparison_{prefix}.png")
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
