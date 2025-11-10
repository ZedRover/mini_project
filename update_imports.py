#!/usr/bin/env python3
"""Update import statements after restructuring"""

import re
from pathlib import Path

# Mapping of old module names to new paths
IMPORT_MAPPING = {
    'data_loader': 'src.s01_data_analysis.data_loader',
    'detailed_eda': 'src.s01_data_analysis.detailed_eda',
    'target_variable_analysis': 'src.s01_data_analysis.target_variable_analysis',
    'unique_values_analysis': 'src.s01_data_analysis.unique_values_analysis',
    'data_prep': 'src.s01_data_analysis.data_prep',

    'cross_validation': 'src.s02_model_training.cross_validation',
    'metrics': 'src.s02_model_training.metrics',
    'model_trainer': 'src.s02_model_training.model_trainer',
    'train_models': 'src.s02_model_training.train_models',

    'lasso_analysis': 'src.s03_hyperparameter_tuning.lasso_analysis',
    'visualization': 'src.s03_hyperparameter_tuning.visualization',

    'feature_selection_comparison': 'src.s04_feature_selection.feature_selection_comparison',
    'lasso_feature_selector': 'src.s04_feature_selection.lasso_feature_selector',
    'lasso_trainer': 'src.s04_feature_selection.lasso_feature_selector',  # renamed
    'lightgbm_feature_selector': 'src.s04_feature_selection.lightgbm_feature_selector',
    'lightgbm_trainer': 'src.s04_feature_selection.lightgbm_feature_selector',  # renamed
    'run_feature_selection': 'src.s04_feature_selection.run_feature_selection',

    # Also update paths with old numeric prefixes
    'src.01_data_analysis': 'src.s01_data_analysis',
    'src.02_model_training': 'src.s02_model_training',
    'src.03_hyperparameter_tuning': 'src.s03_hyperparameter_tuning',
    'src.04_feature_selection': 'src.s04_feature_selection',
    'src.05_results_comparison': 'src.s05_results_comparison',
}

def update_imports_in_file(file_path: Path):
    """Update import statements in a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Update "from X import Y" statements
    for old_module, new_module in IMPORT_MAPPING.items():
        # Pattern: from X import ...
        pattern = rf'\bfrom {re.escape(old_module)} import\b'
        replacement = f'from {new_module} import'
        content = re.sub(pattern, replacement, content)

        # Pattern: import X
        pattern = rf'\bimport {re.escape(old_module)}\b'
        replacement = f'import {new_module}'
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    return False

def main():
    src_dir = Path("src")
    updated_count = 0

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if update_imports_in_file(py_file):
            updated_count += 1

    print(f"\nTotal files updated: {updated_count}")

if __name__ == "__main__":
    main()
