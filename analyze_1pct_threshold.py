#!/usr/bin/env python3
"""
Calcular voxeles que tienen dosis < 1% de la dosis máxima total
"""
import numpy as np
from pathlib import Path

gt = np.load('dose_edep.npy')
D, H, W = gt.shape
total_voxels = D * H * W

# Dosis máxima de todo el volumen
max_dose_total = np.max(gt)

# 1% de la dosis máxima
threshold_1pct = 0.01 * max_dose_total

# Contar voxeles por debajo del threshold
voxels_below_threshold = np.sum(gt < threshold_1pct)
voxels_above_threshold = np.sum(gt >= threshold_1pct)
voxels_zero = np.sum(gt == 0)
voxels_nonzero = np.sum(gt > 0)

print("="*80)
print("ANÁLISIS: Voxeles con dosis < 1% de la dosis máxima")
print("="*80)
print()
print(f"Dosis máxima total: {max_dose_total:.6f}")
print(f"1% de dosis máxima (threshold): {threshold_1pct:.6f}")
print()
print(f"Total de voxeles: {total_voxels:,}")
print()
print(f"Voxeles con dosis < 1% (< {threshold_1pct:.6f}):")
print(f"  Cantidad: {voxels_below_threshold:,}")
print(f"  Porcentaje del volumen total: {100*voxels_below_threshold/total_voxels:.2f}%")
print()
print(f"Voxeles con dosis ≥ 1% (≥ {threshold_1pct:.6f}):")
print(f"  Cantidad: {voxels_above_threshold:,}")
print(f"  Porcentaje del volumen total: {100*voxels_above_threshold/total_voxels:.2f}%")
print()
print("="*80)
print("DESGLOSE ADICIONAL")
print("="*80)
print(f"Voxeles con dosis = 0 (vacío):")
print(f"  Cantidad: {voxels_zero:,}")
print(f"  Porcentaje: {100*voxels_zero/total_voxels:.2f}%")
print()
print(f"Voxeles con dosis > 0 (no vacío):")
print(f"  Cantidad: {voxels_nonzero:,}")
print(f"  Porcentaje: {100*voxels_nonzero/total_voxels:.2f}%")
print()

# De los voxeles > 0, ¿cuántos están < 1%?
voxels_nonzero_below_threshold = np.sum((gt > 0) & (gt < threshold_1pct))
print(f"Voxeles con 0 < dosis < 1% (ruido débil):")
print(f"  Cantidad: {voxels_nonzero_below_threshold:,}")
print(f"  Porcentaje de voxeles no-cero: {100*voxels_nonzero_below_threshold/voxels_nonzero:.2f}%")
print(f"  Porcentaje del volumen total: {100*voxels_nonzero_below_threshold/total_voxels:.2f}%")
print()
print("="*80)
print("INTERPRETACIÓN")
print("="*80)
print()
print(f"Si filtras voxeles con dosis < {threshold_1pct:.6f} (1% de max):")
print(f"  • Eliminas: {voxels_below_threshold:,} voxeles ({100*voxels_below_threshold/total_voxels:.2f}%)")
print(f"  • Mantienes: {voxels_above_threshold:,} voxeles ({100*voxels_above_threshold/total_voxels:.2f}%)")
print()
print(f"De estos voxeles eliminados:")
print(f"  • {voxels_zero:,} son completamente vacío (dosis=0)")
print(f"  • {voxels_nonzero_below_threshold:,} son ruido débil (0<dosis<1%)")
print()
