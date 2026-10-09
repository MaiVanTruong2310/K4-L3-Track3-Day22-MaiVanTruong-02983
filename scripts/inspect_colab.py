import json
import sys

with open('colab/Lab22_DPO_BigGPU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cells summary:")
for i, c in enumerate(nb['cells']):
    outs = c.get('outputs', [])
    if outs:
        types = [o.get('output_type') for o in outs]
        print(f"Cell {i:3d} ({c['cell_type']:8s}) has {len(outs):2d} outputs: {types}")
