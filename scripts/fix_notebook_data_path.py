#!/usr/bin/env python3
"""
Fix notebook data path from /PINN_data to project-relative path.
"""
import json
from pathlib import Path

notebook_path = Path(__file__).parent.parent / "notebooks" / "pinn_2d_forward_validation.ipynb"

# Read notebook
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find and replace data_dir path in code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if isinstance(cell['source'], list):
            source = ''.join(cell['source'])
        else:
            source = cell['source']

        # Replace /PINN_data with project-relative path
        if 'data_dir = Path("/PINN_data")' in source:
            source = source.replace(
                'data_dir = Path("/PINN_data")',
                'data_dir = Path.cwd().parent / "PINN_data"'
            )
            # Update error message
            source = source.replace(
                'Please create /PINN_data/ and place FDTD .npz files there.',
                'Please create PINN_data/ in project root and place FDTD .npz files there.'
            )
            # Convert back to list format
            cell['source'] = source.split('\n')
            # Add newline at end of each line except last
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                            for i, line in enumerate(cell['source'])]
            print(f"Updated cell with data_dir path")

# Also update markdown cell mentioning /PINN_data
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if isinstance(cell['source'], list):
            source = ''.join(cell['source'])
        else:
            source = cell['source']

        if '/PINN_data/' in source:
            source = source.replace('/PINN_data/', 'PINN_data/')
            # Convert back to list format
            cell['source'] = source.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                            for i, line in enumerate(cell['source'])]
            print(f"Updated markdown cell mentioning PINN_data")

# Write updated notebook
with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ Updated {notebook_path}")
