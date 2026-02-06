#!/usr/bin/env python3
"""
Fix all .mhd files to point to correct .raw files
"""

from pathlib import Path
import re

DATASET_ROOT = Path("dataset_pilot")

print("🔧 Arreglando referencias en archivos .mhd...")

for mhd_file in sorted(DATASET_ROOT.rglob("*.mhd")):
    # Get the base name (input_1M, target, etc)
    base_name = mhd_file.stem  # Without .mhd extension
    raw_file = mhd_file.with_name(f"{base_name}.raw")
    
    # Read the .mhd file
    content = mhd_file.read_text()
    
    # Replace ElementDataFile line
    new_content = re.sub(
        r"ElementDataFile = .*\.raw",
        f"ElementDataFile = {base_name}.raw",
        content
    )
    
    # Write back if changed
    if new_content != content:
        mhd_file.write_text(new_content)
        print(f"✓ {mhd_file.relative_to(DATASET_ROOT)}")
        
        # Verify
        updated = mhd_file.read_text()
        for line in updated.split('\n'):
            if 'ElementDataFile' in line:
                print(f"  → {line.strip()}")

print("\n✅ Listo. Verifica que los .raw existan:")
for mhd_file in list(DATASET_ROOT.rglob("*.mhd"))[:5]:
    base_name = mhd_file.stem
    raw_file = mhd_file.with_name(f"{base_name}.raw")
    exists = "✓" if raw_file.exists() else "✗"
    print(f"{exists} {raw_file.relative_to(DATASET_ROOT)}")
