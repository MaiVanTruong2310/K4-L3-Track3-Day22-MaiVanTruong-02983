import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('colab/Lab22_DPO_BigGPU.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [44, 45, 54, 57, 66, 68, 69, 71, 78, 80, 81, 82, 83, 84, 87, 89, 104, 106, 107]:
    c = nb['cells'][idx]
    print(f"==================== CELL {idx} ====================")
    for o in c.get('outputs', []):
        if 'text' in o:
            print(''.join(o['text']))
        if 'data' in o and 'text/plain' in o['data']:
            print(''.join(o['data']['text/plain']))
