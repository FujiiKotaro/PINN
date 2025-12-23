#!/usr/bin/env python3
"""
Clean invalid metadata from notebook cells.

Removes 'execution_count' and 'outputs' properties from markdown cells,
which should only exist in code cells.
"""
import json
from pathlib import Path

notebook_path = Path(__file__).parent.parent / "notebooks" / "pinn_2d_forward_validation.ipynb"

# Read notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

cleaned_cells = 0

# Clean markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        # Remove invalid properties that should only exist in code cells
        removed_props = []

        if 'execution_count' in cell:
            del cell['execution_count']
            removed_props.append('execution_count')

        if 'outputs' in cell:
            del cell['outputs']
            removed_props.append('outputs')

        if removed_props:
            cleaned_cells += 1
            cell_id = cell.get('id', 'unknown')
            print(f"Cleaned cell {cell_id}: removed {', '.join(removed_props)}")

# Write updated notebook
if cleaned_cells > 0:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Cleaned {cleaned_cells} markdown cell(s) in {notebook_path}")
else:
    print(f"✓ No invalid metadata found in {notebook_path}")
