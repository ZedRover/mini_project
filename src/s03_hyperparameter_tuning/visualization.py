#!/usr/bin/env python3
"""
可视化模块
生成各种图表用于模型分析和结果展示
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径，支持 uv run 和 python -m 两种运行方式
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.utils.cross_validation import CrossValidationResult

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:
    """可视化工具类"""

    def __init__(self, output_dir: str | Path = "results/figures"):
        """
        初始化可视化工具

        Parameters
        ----------
        output_dir : str | Path
            图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_lasso_ic_heatmap(
        self,
        ic_matrix: pd.DataFrame,
        title: str = "LASSO: IC across Alphas and Folds",
        filename: str = "lasso_ic_heatmap.png"
    ) -> None:
        """
        绘制LASSO不同alpha和fold的IC热力图

        Parameters
        ----------
        ic_matrix : pd.DataFrame
            alpha × fold 的IC矩阵
        title : str
            图表标题
        filename : str
            保存的文件名
        """
        fig, ax = plt.subplots(figsize=(10, max(8, len(ic_matrix) * 0.4)))

        # 创建热力图
        sns.heatmap(
            ic_matrix,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "IC"},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Alpha", fontsize=12)

        # 格式化y轴标签（alpha值）
        y_labels = [f"{float(label):.2e}" for label in ic_matrix.index]
        ax.set_yticklabels(y_labels, rotation=0)

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def plot_lasso_stability(
        self,
        stability_df: pd.DataFrame,
        filename: str = "lasso_stability.png"
    ) -> None:
        """
        绘制LASSO不同alpha的稳定性分析图

        Parameters
        ----------
        stability_df : pd.DataFrame
            稳定性指标DataFrame
        filename : str
            保存的文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. IC均值 vs alpha
        ax = axes[0, 0]
        ax.plot(stability_df["alpha"], stability_df["ic_mean"], marker="o", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (log scale)", fontsize=11)
        ax.set_ylabel("Mean IC", fontsize=11)
        ax.set_title("IC Mean vs Alpha", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 2. IC标准差 vs alpha
        ax = axes[0, 1]
        ax.plot(stability_df["alpha"], stability_df["ic_std"], marker="o", color="orange", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (log scale)", fontsize=11)
        ax.set_ylabel("IC Std Dev", fontsize=11)
        ax.set_title("IC Stability vs Alpha", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 3. IC变异系数 vs alpha
        ax = axes[1, 0]
        # 过滤掉无穷大的值
        valid_cv = stability_df[np.isfinite(stability_df["ic_cv"])]
        ax.plot(valid_cv["alpha"], valid_cv["ic_cv"], marker="o", color="red", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (log scale)", fontsize=11)
        ax.set_ylabel("Coefficient of Variation", fontsize=11)
        ax.set_title("IC CV vs Alpha", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 4. IC范围 vs alpha
        ax = axes[1, 1]
        ax.plot(stability_df["alpha"], stability_df["ic_range"], marker="o", color="green", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Alpha (log scale)", fontsize=11)
        ax.set_ylabel("IC Range (Max - Min)", fontsize=11)
        ax.set_title("IC Range vs Alpha", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.suptitle("LASSO Stability Analysis Across Different Alphas", fontsize=14, y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def plot_model_comparison(
        self,
        cv_results: List[CrossValidationResult],
        metrics: List[str] = None,
        filename: str = "model_comparison.png"
    ) -> None:
        """
        绘制多个模型的对比图

        Parameters
        ----------
        cv_results : List[CrossValidationResult]
            多个模型的交叉验证结果
        metrics : List[str], optional
            要对比的指标列表
        filename : str
            保存的文件名
        """
        if metrics is None:
            metrics = ["ic_pearson", "ic_spearman", "rmse", "r2"]

        model_names = [cv.model_name for cv in cv_results]
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # 收集每个模型的均值和标准差
            means = []
            stds = []
            for cv_result in cv_results:
                mean_key = f"val_{metric}_mean"
                std_key = f"val_{metric}_std"
                if mean_key in cv_result.aggregate_metrics:
                    means.append(cv_result.aggregate_metrics[mean_key])
                    stds.append(cv_result.aggregate_metrics[std_key])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # 绘制柱状图
            x_pos = np.arange(len(model_names))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue")

            # 在柱子上标注数值
            for i, (m, s) in enumerate(zip(means, stds)):
                if not np.isnan(m):
                    ax.text(i, m, f"{m:.4f}", ha="center", va="bottom", fontsize=9)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            ax.set_ylabel(f"{metric.upper()}", fontsize=11)
            ax.set_title(f"Validation {metric.upper()}", fontsize=12)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Model Comparison (Cross-Validation)", fontsize=14, y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def plot_ic_boxplot(
        self,
        cv_results: List[CrossValidationResult],
        filename: str = "ic_boxplot.png"
    ) -> None:
        """
        绘制多个模型IC分布的箱线图

        Parameters
        ----------
        cv_results : List[CrossValidationResult]
            多个模型的交叉验证结果
        filename : str
            保存的文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 准备数据
        data_pearson = []
        data_spearman = []
        model_names = []

        for cv_result in cv_results:
            model_names.append(cv_result.model_name)

            # 收集每个fold的IC
            pearson_ics = [fr.val_metrics["ic_pearson"] for fr in cv_result.fold_results]
            spearman_ics = [fr.val_metrics["ic_spearman"] for fr in cv_result.fold_results]

            data_pearson.append(pearson_ics)
            data_spearman.append(spearman_ics)

        # Pearson IC 箱线图
        ax = axes[0]
        bp = ax.boxplot(data_pearson, labels=model_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_ylabel("IC (Pearson)", fontsize=12)
        ax.set_title("Pearson IC Distribution", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Spearman IC 箱线图
        ax = axes[1]
        bp = ax.boxplot(data_spearman, labels=model_names, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightcoral")
        ax.set_ylabel("IC (Spearman)", fontsize=12)
        ax.set_title("Spearman IC Distribution", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle("IC Distribution Across Folds", fontsize=14, y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def plot_quantile_performance(
        self,
        cv_results: List[CrossValidationResult],
        quantiles: List[str] = None,
        filename: str = "quantile_performance.png"
    ) -> None:
        """
        绘制不同模型在不同分位数的表现

        Parameters
        ----------
        cv_results : List[CrossValidationResult]
            多个模型的交叉验证结果
        quantiles : List[str], optional
            要展示的分位数标签
        filename : str
            保存的文件名
        """
        if quantiles is None:
            quantiles = [
                "bottom_10%_mean", "bottom_5%_mean", "bottom_1%_mean",
                "top_10%_mean", "top_5%_mean", "top_1%_mean"
            ]

        model_names = [cv.model_name for cv in cv_results]

        # 收集数据
        data = []
        for cv_result in cv_results:
            row = {"model": cv_result.model_name}
            for q in quantiles:
                key = f"val_{q}_mean"
                if key in cv_result.aggregate_metrics:
                    row[q] = cv_result.aggregate_metrics[key]
                else:
                    row[q] = np.nan
            data.append(row)

        df = pd.DataFrame(data)

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bottom quantiles
        ax = axes[0]
        bottom_cols = [q for q in quantiles if "bottom" in q]
        df_bottom = df[["model"] + bottom_cols].set_index("model")
        df_bottom.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Bottom Quantile Performance (Mean realY)", fontsize=12)
        ax.set_ylabel("Mean realY", fontsize=11)
        ax.set_xlabel("")
        ax.legend(title="Quantile", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Top quantiles
        ax = axes[1]
        top_cols = [q for q in quantiles if "top" in q]
        df_top = df[["model"] + top_cols].set_index("model")
        df_top.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Top Quantile Performance (Mean realY)", fontsize=12)
        ax.set_ylabel("Mean realY", fontsize=11)
        ax.set_xlabel("")
        ax.legend(title="Quantile", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle("Quantile Performance Analysis", fontsize=14, y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")

    def plot_prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        filename: str = "prediction_scatter.png"
    ) -> None:
        """
        绘制预测值vs真实值的散点图

        Parameters
        ----------
        y_true : np.ndarray
            真实值
        y_pred : np.ndarray
            预测值
        model_name : str
            模型名称
        filename : str
            保存的文件名
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制散点图
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)

        # 绘制对角线
        lims = [
            np.min([y_true.min(), y_pred.min()]),
            np.max([y_true.max(), y_pred.max()])
        ]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")

        ax.set_xlabel("True Value", fontsize=12)
        ax.set_ylabel("Predicted Value", fontsize=12)
        ax.set_title(f"{model_name}: Prediction vs True Value", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {filename}")


if __name__ == "__main__":
    # 测试可视化工具
    viz = Visualizer(output_dir="test_figures")

    # 创建测试数据
    ic_matrix = pd.DataFrame(
        np.random.randn(5, 4) * 0.1 + 0.3,
        index=[0.001, 0.01, 0.1, 1.0, 10.0],
        columns=["fold_1", "fold_2", "fold_3", "fold_4"]
    )

    stability_df = pd.DataFrame({
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "ic_mean": [0.35, 0.38, 0.40, 0.37, 0.30],
        "ic_std": [0.05, 0.04, 0.03, 0.04, 0.06],
        "ic_cv": [0.14, 0.11, 0.08, 0.11, 0.20],
        "ic_range": [0.12, 0.10, 0.08, 0.10, 0.15],
    })

    viz.plot_lasso_ic_heatmap(ic_matrix, filename="test_heatmap.png")
    viz.plot_lasso_stability(stability_df, filename="test_stability.png")

    print("\n测试可视化完成！")
