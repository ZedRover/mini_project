#!/usr/bin/env python3
"""
交叉验证框架
实现K-Fold交叉验证，记录每个fold的训练和验证结果
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

# 添加项目根目录到Python路径，支持 uv run 和 python -m 两种运行方式
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, TimeSeriesSplit

from src.utils.metrics import MetricsCalculator


@dataclass
class FoldResult:
    """单个fold的结果"""
    fold_idx: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    train_time: float
    model: BaseEstimator | None = None


@dataclass
class CrossValidationResult:
    """交叉验证完整结果"""
    model_name: str
    n_folds: int
    fold_results: List[FoldResult]
    aggregate_metrics: Dict[str, float]


class CrossValidator:
    """交叉验证器"""

    def __init__(
        self,
        n_folds: int = 4,
        random_state: int = 42,
        shuffle: bool = True,
        use_time_series_split: bool = True,
        metrics_calculator: MetricsCalculator | None = None
    ):
        """
        初始化交叉验证器

        Parameters
        ----------
        n_folds : int
            折数，默认为4
        random_state : int
            随机种子（仅当use_time_series_split=False时使用）
        shuffle : bool
            是否在划分前打乱数据（仅当use_time_series_split=False时使用）
        use_time_series_split : bool
            是否使用时序交叉验证。True=使用TimeSeriesSplit（防止数据泄露），
            False=使用KFold（可能导致时序数据泄露）
        metrics_calculator : MetricsCalculator, optional
            指标计算器，如果为None则创建默认的
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.shuffle = shuffle
        self.use_time_series_split = use_time_series_split

        if use_time_series_split:
            # 使用时序交叉验证（不shuffle，保持时序）
            self.splitter = TimeSeriesSplit(n_splits=n_folds)
            self.kfold = None  # 保持向后兼容
        else:
            # 使用标准KFold（可能导致时序数据泄露）
            self.kfold = KFold(
                n_splits=n_folds,
                shuffle=shuffle,
                random_state=random_state
            )
            self.splitter = self.kfold

        self.metrics_calculator = metrics_calculator or MetricsCalculator()

    def run_cv(
        self,
        model: BaseEstimator,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        model_name: str = "Model",
        verbose: bool = True,
        return_models: bool = False
    ) -> CrossValidationResult:
        """
        运行交叉验证

        Parameters
        ----------
        model : BaseEstimator
            要训练的模型
        X : pd.DataFrame | np.ndarray
            特征矩阵
        y : pd.Series | np.ndarray
            目标变量
        model_name : str
            模型名称
        verbose : bool
            是否打印详细信息
        return_models : bool
            是否保存训练好的模型

        Returns
        -------
        CrossValidationResult
            交叉验证结果
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"{model_name} - {self.n_folds}-Fold 交叉验证")
            print("=" * 70)

        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(X), 1):
            if verbose:
                print(f"\n[Fold {fold_idx}/{self.n_folds}]")

            # 划分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if verbose:
                print(f"  训练集: {len(X_train):,} 样本")
                print(f"  验证集: {len(X_val):,} 样本")

            # 训练模型
            import time
            start_time = time.time()

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            train_time = time.time() - start_time

            # 预测
            y_train_pred = fold_model.predict(X_train)
            y_val_pred = fold_model.predict(X_val)

            # 计算指标
            train_metrics = self.metrics_calculator.compute_all_metrics(
                y_train, y_train_pred, include_quantile=False
            )
            val_metrics = self.metrics_calculator.compute_all_metrics(
                y_val, y_val_pred, include_quantile=True
            )

            if verbose:
                print(f"  训练时间: {train_time:.2f}s")
                print(f"  训练集 IC: {train_metrics['ic_pearson']:.6f}")
                print(f"  验证集 IC: {val_metrics['ic_pearson']:.6f}")
                print(f"  验证集 RMSE: {val_metrics['rmse']:.6f}")

            # 保存结果
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_time=train_time,
                model=fold_model if return_models else None
            )
            fold_results.append(fold_result)

        # 计算聚合指标
        aggregate_metrics = self._aggregate_metrics(fold_results)

        if verbose:
            print("\n" + "-" * 70)
            print("交叉验证汇总")
            print("-" * 70)
            print(f"平均验证集 IC (Pearson):  {aggregate_metrics['val_ic_pearson_mean']:.6f} ± {aggregate_metrics['val_ic_pearson_std']:.6f}")
            print(f"平均验证集 IC (Spearman): {aggregate_metrics['val_ic_spearman_mean']:.6f} ± {aggregate_metrics['val_ic_spearman_std']:.6f}")
            print(f"平均验证集 RMSE:          {aggregate_metrics['val_rmse_mean']:.6f} ± {aggregate_metrics['val_rmse_std']:.6f}")
            print(f"平均训练时间:            {aggregate_metrics['train_time_mean']:.2f}s")
            print("=" * 70)

        return CrossValidationResult(
            model_name=model_name,
            n_folds=self.n_folds,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics
        )

    def _aggregate_metrics(self, fold_results: List[FoldResult]) -> Dict[str, float]:
        """
        聚合多个fold的指标

        Parameters
        ----------
        fold_results : List[FoldResult]
            所有fold的结果

        Returns
        -------
        Dict[str, float]
            聚合后的指标（均值和标准差）
        """
        aggregate = {}

        # 收集所有验证集指标
        val_metrics_list = [fr.val_metrics for fr in fold_results]
        train_metrics_list = [fr.train_metrics for fr in fold_results]
        train_times = [fr.train_time for fr in fold_results]

        # 计算验证集指标的均值和标准差
        metric_keys = val_metrics_list[0].keys()
        for key in metric_keys:
            values = [m[key] for m in val_metrics_list]
            aggregate[f"val_{key}_mean"] = float(np.mean(values))
            aggregate[f"val_{key}_std"] = float(np.std(values))

        # 计算训练集指标的均值和标准差
        for key in train_metrics_list[0].keys():
            values = [m[key] for m in train_metrics_list]
            aggregate[f"train_{key}_mean"] = float(np.mean(values))
            aggregate[f"train_{key}_std"] = float(np.std(values))

        # 训练时间统计
        aggregate["train_time_mean"] = float(np.mean(train_times))
        aggregate["train_time_std"] = float(np.std(train_times))
        aggregate["train_time_total"] = float(np.sum(train_times))

        return aggregate

    def export_fold_details(
        self,
        cv_result: CrossValidationResult,
        output_path: str | None = None
    ) -> pd.DataFrame:
        """
        导出每个fold的详细指标

        Parameters
        ----------
        cv_result : CrossValidationResult
            交叉验证结果
        output_path : str, optional
            输出CSV文件路径

        Returns
        -------
        pd.DataFrame
            包含所有fold详细指标的DataFrame
        """
        rows = []
        for fr in cv_result.fold_results:
            row = {
                "model": cv_result.model_name,
                "fold": fr.fold_idx,
                "train_time": fr.train_time,
            }
            # 添加训练集指标（加前缀）
            for key, val in fr.train_metrics.items():
                row[f"train_{key}"] = val
            # 添加验证集指标（加前缀）
            for key, val in fr.val_metrics.items():
                row[f"val_{key}"] = val
            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"详细fold结果已保存至: {output_path}")

        return df


def compare_cv_results(
    cv_results: List[CrossValidationResult],
    metric_key: str = "val_ic_pearson_mean"
) -> pd.DataFrame:
    """
    比较多个模型的交叉验证结果

    Parameters
    ----------
    cv_results : List[CrossValidationResult]
        多个模型的交叉验证结果
    metric_key : str
        用于排序的指标键

    Returns
    -------
    pd.DataFrame
        模型对比表格
    """
    rows = []
    for cv_result in cv_results:
        row = {
            "model": cv_result.model_name,
            "n_folds": cv_result.n_folds,
            **cv_result.aggregate_metrics
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 按指定指标排序
    if metric_key in df.columns:
        df = df.sort_values(metric_key, ascending=False)

    return df


if __name__ == "__main__":
    # 测试交叉验证
    from sklearn.linear_model import Ridge

    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(1000, 50)
    y = X[:, 0] + X[:, 1] * 2 + np.random.randn(1000) * 0.5

    # 运行交叉验证
    cv = CrossValidator(n_folds=4, random_state=42)
    model = Ridge(alpha=1.0)
    result = cv.run_cv(model, X, y, model_name="Ridge", verbose=True)

    # 导出详细结果
    df = cv.export_fold_details(result)
    print("\n详细fold结果:")
    print(df.head())
