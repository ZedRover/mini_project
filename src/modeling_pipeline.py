#!/usr/bin/env python3
"""Model comparison script with IC and top-quantile return analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "realY"
RANDOM_STATE = 42
DATA_PATH = Path("data.csv")
RESULTS_PATH = Path("model_comparison_results.csv")


@dataclass
class ModelReport:
    name: str
    metrics: Dict[str, float]


def load_dataset(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col=0)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not present in dataset.")
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_pred.std() == 0 or y_true.std() == 0:
        return {"ic_pearson": 0.0, "ic_spearman": 0.0}

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    y_true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1])
    return {"ic_pearson": pearson, "ic_spearman": spearman}


def compute_top_bucket_returns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Iterable[int] = (90, 99),
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for threshold in thresholds:
        cutoff = np.percentile(y_pred, threshold)
        mask = y_pred >= cutoff
        coverage_pct = mask.mean() * 100
        mean_return = float(np.mean(y_true[mask])) if mask.sum() > 0 else float("nan")
        stats[f"top_{threshold}_coverage_pct"] = coverage_pct
        stats[f"top_{threshold}_mean_return"] = mean_return
    return stats


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "rmse": mse**0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    metrics.update(compute_information_coefficient(y_true, y_pred))
    metrics.update(compute_top_bucket_returns(y_true, y_pred))
    return metrics


def get_models() -> Dict[str, Pipeline | LGBMRegressor]:
    return {
        "LinearRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=1e-3, max_iter=10_000, random_state=RANDOM_STATE)),
            ]
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "NeuralNetwork": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        learning_rate_init=1e-3,
                        batch_size=256,
                        max_iter=400,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


def run_training_suite(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> List[ModelReport]:
    models = get_models()
    reports: List[ModelReport] = []

    for name, estimator in models.items():
        print(f"\n=== Training {name} ===")
        start = perf_counter()
        estimator.fit(X_train, y_train)
        duration = perf_counter() - start
        y_pred = estimator.predict(X_test)
        metrics = evaluate_predictions(y_test.to_numpy(), y_pred)
        metrics["fit_seconds"] = duration
        reports.append(ModelReport(name=name, metrics=metrics))
        print(f"{name} completed in {duration:.2f}s | IC (Pearson): {metrics['ic_pearson']:.4f}")

    return reports


def summarize_and_save(reports: List[ModelReport]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"model": report.name, **report.metrics}
            for report in reports
        ]
    )
    df = df.sort_values(by="ic_pearson", ascending=False)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved detailed metrics to {RESULTS_PATH.resolve()}")
    print("\nModel comparison (sorted by IC):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    return df


def main() -> None:
    print("Loading dataset...")
    X, y = load_dataset(DATA_PATH, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    reports = run_training_suite(X_train, X_test, y_train, y_test)
    summarize_and_save(reports)


if __name__ == "__main__":
    main()
