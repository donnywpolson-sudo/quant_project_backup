import json
import sys
import argparse
from pathlib import Path

def validate(manifest_path: str):
    path = Path(manifest_path)
    if not path.exists():
        print(f'Error: Manifest not found at {manifest_path}')
        sys.exit(1)
    with open(path, 'r') as f:
        data = json.load(f)
    required_keys = {'feature_names', 'dtypes', 'selection_seed', 'selection_date', 'selection_model', 'selection_params', 'selected_K', 'cumulative_importance', 'stability_stats', 'baseline_feature_list', 'baseline_features_hash', 'baseline_feature_matrix_path', 'serialization_params', 'discovery_status', 'folds', 'htf_features_included'}
    missing = required_keys - set(data.keys())
    if missing:
        print(f'Validation Failed: Missing keys: {missing}')
        sys.exit(1)
    print(f'Manifest at {manifest_path} is structurally compliant.')
    sys.exit(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Path to manifest.json')
    args = parser.parse_args()
    validate(args.path)