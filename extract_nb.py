import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = r'd:\Dosen\0 Genap 2026\Penelitian\Pak Chindra\irigasi_ai'
notebooks = [
    '01_preprocessing_sensor.ipynb',
    '02_preprocessing.ipynb',
    '03_random_forest.ipynb',
    '04_xgboost.ipynb',
    '05_evaluasi_model.ipynb',
]

for nb_name in notebooks:
    path = os.path.join(base, nb_name)
    print(f'\n\n{"="*60}')
    print(f'NOTEBOOK: {nb_name}')
    print(f'{"="*60}')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        ct = cell['cell_type']
        src = ''.join(cell['source'])
        if ct == 'code' and src.strip():
            print(f'\n--- CELL {i} ({ct}) ---')
            print(src[:3000])
