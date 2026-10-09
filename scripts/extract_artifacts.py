import json
import base64
from pathlib import Path

with open('colab/Lab22_DPO_BigGPU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Save images
screenshots_dir = Path('submission/screenshots')
screenshots_dir.mkdir(parents=True, exist_ok=True)

for i in [42, 55, 68]:
    cell = nb['cells'][i]
    for out in cell.get('outputs', []):
        if 'data' in out:
            for mime, val in out['data'].items():
                if mime == 'image/png':
                    img_data = base64.b64decode(''.join(val))
                    if i == 42:
                        target = screenshots_dir / '02-sft-loss.png'
                    elif i == 55:
                        target = screenshots_dir / '02b-pref-length.png'
                    elif i == 68:
                        target = screenshots_dir / '03-dpo-reward-curves.png'
                    else:
                        target = screenshots_dir / f"cell_{i}.png"
                    target.write_bytes(img_data)
                    print(f"Saved {target} ({len(img_data)} bytes)")

# Check cell 81, 82, 83
for idx in [81, 82, 83]:
    if idx < len(nb['cells']):
        c = nb['cells'][idx]
        print(f"Cell {idx} ({c['cell_type']}):")
        print(''.join(c.get('source', [])))
