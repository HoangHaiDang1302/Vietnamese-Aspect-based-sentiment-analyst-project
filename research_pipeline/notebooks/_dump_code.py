import json

def get_code(file_path):
    print(f"--- {file_path} ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            if 'import' in src or 'load' in src or 'predict' in src or 'Dataset' in src:
                print(src[:200] + "\n...")
    print("\n")

get_code('research_pipeline/notebooks/03-joint-kaggle.ipynb')
get_code('research_pipeline/notebooks/04-e2e-kaggle.ipynb')
get_code('research_pipeline/notebooks/05-e2e-bigru-baseline.ipynb')
