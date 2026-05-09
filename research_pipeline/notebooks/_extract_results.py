import json
import os

notebooks = [
    'research_pipeline/notebooks/01-ate-kaggle.ipynb',
    'research_pipeline/notebooks/02-asc-kaggle.ipynb',
    'research_pipeline/notebooks/03-joint-kaggle.ipynb',
    'research_pipeline/notebooks/04-e2e-kaggle.ipynb',
    'research_pipeline/notebooks/05-e2e-bigru-baseline.ipynb'
]

results = {}

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        results[nb_path] = "Not found"
        continue
        
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    print(f"\n{'='*20} {os.path.basename(nb_path)} {'='*20}")
    # Search for cells with "Evaluation" or final results
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out.get('output_type') == 'stream' and 'text' in out:
                    text = ''.join(out['text'])
                    # Filter for keywords like F1, Precision, Recall, Accuracy
                    if any(kw in text for kw in ['F1', 'Precision', 'Recall', 'Accuracy']):
                        # Only show relevant summary lines
                        lines = [l for l in text.split('\n') if any(kw in l for kw in ['F1', 'Precision', 'Recall', 'Accuracy', 'BEST', 'Model'])]
                        if lines:
                            print('\n'.join(lines[:10])) # limit output
                elif out.get('output_type') == 'display_data' or out.get('output_type') == 'execute_result':
                    # Sometimes results are in DataFrames (HTML)
                    pass
