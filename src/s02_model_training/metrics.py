#!/usr/bin/env python3
"""
评估指标模块
包含回归指标、IC指标、分位数IC分析等
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, quantiles: List[int] = None):
        """
        初始化指标计算器

        Parameters
        ----------
        quantiles : List[int], optional
            用于分位数分析的百分位数，默认为 [1, 5, 10, 90, 95, 99]
        """
        if quantiles is None:
            quantiles = [1, 5, 10, 90, 95, 99]
        self.quantiles = quantiles

    def compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算基础回归指标

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值

        Returns
        -------
        Dict[str, float]
            包含 RMSE, MAE, R² 的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
        }

    def compute_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算信息系数（Information Coefficient）

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值

        Returns
        -------
        Dict[str, float]
            包含 Pearson IC 和 Spearman IC 的字典
        """
        # 检查标准差是否为0
        if y_pred.std() == 0 or y_true.std() == 0:
            return {"ic_pearson": 0.0, "ic_spearman": 0.0}

        # Pearson 相关系数
        pearson_ic, _ = pearsonr(y_true, y_pred)

        # Spearman 秩相关系数
        spearman_ic, _ = spearmanr(y_true, y_pred)

        return {
            "ic_pearson": float(pearson_ic),
            "ic_spearman": float(spearman_ic),
        }

    def compute_quantile_ic(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算不同分位数的IC（基于预测值排序）

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值

        Returns
        -------
        Dict[str, float]
            包含各分位数IC的字典
        """
        results = {}

        for q in self.quantiles:
            if q < 50:  # Bottom quantiles
                # 计算底部分位数的IC
                threshold = np.percentile(y_pred, q)
                mask = y_pred <= threshold
                label = f"bottom_{q}%"
            else:  # Top quantiles
                # 计算顶部分位数的IC
                threshold = np.percentile(y_pred, q)
                mask = y_pred >= threshold
                label = f"top_{100-q}%"

            # 提取对应分位数的样本
            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]

            # 计算IC
            if len(y_true_subset) > 1 and y_pred_subset.std() > 0 and y_true_subset.std() > 0:
                ic_pearson, _ = pearsonr(y_true_subset, y_pred_subset)
                ic_spearman, _ = spearmanr(y_true_subset, y_pred_subset)
                results[f"{label}_ic_pearson"] = float(ic_pearson)
                results[f"{label}_ic_spearman"] = float(ic_spearman)
            else:
                results[f"{label}_ic_pearson"] = 0.0
                results[f"{label}_ic_spearman"] = 0.0

        return results

    def compute_quantile_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算不同分位数的真实值表现（基于预测值排序）

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值

        Returns
        -------
        Dict[str, float]
            包含各分位数真实值均值和标准差的字典
        """
        results = {}

        for q in self.quantiles:
            if q < 50:  # Bottom quantiles
                threshold = np.percentile(y_pred, q)
                mask = y_pred <= threshold
                label = f"bottom_{q}%"
            else:  # Top quantiles
                threshold = np.percentile(y_pred, q)
                mask = y_pred >= threshold
                label = f"top_{100-q}%"

            # 提取对应分位数的真实值
            y_true_subset = y_true[mask]

            if len(y_true_subset) > 0:
                results[f"{label}_mean"] = float(np.mean(y_true_subset))
                results[f"{label}_std"] = float(np.std(y_true_subset))
                results[f"{label}_count"] = int(len(y_true_subset))
            else:
                results[f"{label}_mean"] = np.nan
                results[f"{label}_std"] = np.nan
                results[f"{label}_count"] = 0

        return results

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        include_quantile: bool = True
    ) -> Dict[str, float]:
        """
        计算所有评估指标

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值
        include_quantile : bool, optional
            是否包含分位数分析，默认为 True

        Returns
        -------
        Dict[str, float]
            包含所有指标的字典
        """
        metrics = {}

        # 基础指标
        metrics.update(self.compute_basic_metrics(y_true, y_pred))

        # IC指标
        metrics.update(self.compute_ic(y_true, y_pred))

        # 分位数分析
        if include_quantile:
            metrics.update(self.compute_quantile_ic(y_true, y_pred))
            metrics.update(self.compute_quantile_performance(y_true, y_pred))

        return metrics

    def format_metrics_table(self, metrics: Dict[str, float]) -> pd.DataFrame:
        """
        将指标格式化为易读的表格

        Parameters
        ----------
        metrics : Dict[str, float]
            指标字典

        Returns
        -------
        pd.DataFrame
            格式化后的指标表格
        """
        df = pd.DataFrame([metrics]).T
        df.columns = ["Value"]
        df.index.name = "Metric"
        return df

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        比较多个模型的指标

        Parameters
        ----------
        results : Dict[str, Dict[str, float]]
            模型名称到指标字典的映射

        Returns
        -------
        pd.DataFrame
            模型对比表格
        """
        df = pd.DataFrame(results).T
        df.index.name = "Model"
        return df


def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """
    打印指标摘要

    Parameters
    ----------
    metrics : Dict[str, float]
        指标字典
    model_name : str
        模型名称
    """
    print("\n" + "=" * 70)
    print(f"{model_name} - 评估指标")
    print("=" * 70)

    # 基础指标
    print("\n【基础指标】")
    print(f"  RMSE:       {metrics.get('rmse', np.nan):.6f}")
    print(f"  MAE:        {metrics.get('mae', np.nan):.6f}")
    print(f"  R²:         {metrics.get('r2', np.nan):.6f}")

    # IC指标
    print("\n【IC指标】")
    print(f"  Pearson:    {metrics.get('ic_pearson', np.nan):.6f}")
    print(f"  Spearman:   {metrics.get('ic_spearman', np.nan):.6f}")

    # 分位数IC
    print("\n【分位数IC - Pearson】")
    for q in [1, 5, 10]:
        key = f"bottom_{q}%_ic_pearson"
        if key in metrics:
            print(f"  Bottom {q}%:  {metrics[key]:.6f}")
    for q in [10, 5, 1]:
        key = f"top_{q}%_ic_pearson"
        if key in metrics:
            print(f"  Top {q}%:     {metrics[key]:.6f}")

    # 分位数表现
    print("\n【分位数真实值均值】")
    for q in [1, 5, 10]:
        key = f"bottom_{q}%_mean"
        if key in metrics:
            print(f"  Bottom {q}%:  {metrics[key]:.6f}")
    for q in [10, 5, 1]:
        key = f"top_{q}%_mean"
        if key in metrics:
            print(f"  Top {q}%:     {metrics[key]:.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 测试指标计算
    np.random.seed(42)
    y_true = np.random.randn(1000)
    y_pred = y_true + np.random.randn(1000) * 0.5

    calculator = MetricsCalculator()
    metrics = calculator.compute_all_metrics(y_true, y_pred)

    print_metrics_summary(metrics, "Test Model")
