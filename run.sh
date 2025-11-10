uv run src/s01_data_analysis/target_variable_analysis.py
uv run src/s01_data_analysis/unique_values_analysis.py

uv run src/s02_model_training/train_models.py

uv run src/s03_hyperparameter_tuning/lasso_analysis.py

uv run src/s03_hyperparameter_tuning/lightgbm_tuning.py
uv run src/s03_hyperparameter_tuning/visualization.py
uv run src/s04_feature_selection/run_feature_selection.py