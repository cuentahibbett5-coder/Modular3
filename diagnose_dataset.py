#!/usr/bin/env python3
"""
Diagnóstico: Ver qué está buscando el dataset
"""

from pathlib import Path

DATASET_ROOT = Path("dataset_pilot")

print("🔍 DIAGNÓSTICO DEL DATASET")
print("=" * 60)

# Ver targets
print("\n📍 Targets encontrados:")
targets = sorted(DATASET_ROOT.glob("target_*"))
for t in targets:
    files = list(t.glob("*"))
    print(f"  {t.name}: {[f.name for f in files]}")

if not targets:
    print("  ❌ NO HAY TARGETS")

print("\n📍 Split val:")
val_dir = DATASET_ROOT / "val"
if not val_dir.exists():
    print("  ❌ NO EXISTE val/")
else:
    pairs = sorted(val_dir.glob("pair_*"))
    print(f"  Total pairs: {len(pairs)}")
    
    if len(pairs) > 0:
        # Analizar primer pair
        pair = pairs[0]
        print(f"\n  Ejemplo: {pair.name}")
        print(f"    Archivos directamente:")
        for f in sorted(pair.glob("*")):
            if f.is_file():
                print(f"      - {f.name}")
        
        print(f"    Subdirectorios:")
        for d in sorted(pair.glob("*")):
            if d.is_dir():
                print(f"      {d.name}/")
                for f in sorted(d.glob("*")):
                    if f.is_file():
                        print(f"        - {f.name}")

print("\n📍 Lo que busca el dataset:")
print("  Targets: dataset_pilot/target_N/dose_edep.mhd")
print("  Inputs (opción 1): pair_XXX/input_1M.mhd")
print("  Inputs (opción 2): pair_XXX/input_1M/dose_edep.mhd")

print("\n🔎 Verificando archivos que debería encontrar:")
levels = ["input_1M", "input_2M", "input_5M", "input_10M"]
pair_021 = DATASET_ROOT / "val" / "pair_021"

if pair_021.exists():
    print(f"\n  En {pair_021}:")
    for level in levels:
        mhd_file = pair_021 / f"{level}.mhd"
        dir_file = pair_021 / level / "dose_edep.mhd"
        
        exists_mhd = "✓" if mhd_file.exists() else "✗"
        exists_dir = "✓" if dir_file.exists() else "✗"
        
        print(f"    {level}:")
        print(f"      {level}.mhd: {exists_mhd}")
        print(f"      {level}/dose_edep.mhd: {exists_dir}")
else:
    print(f"  ❌ {pair_021} no existe")

print("\n" + "=" * 60)
