#!/usr/bin/env python3
"""
LASSO超参数分析模块
分析不同alpha值对不同fold的IC表现影响
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到Python路径，支持 uv run 和 python -m 两种运行方式
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.s02_model_training.cross_validation import CrossValidationResult, CrossValidator
from src.s02_model_training.model_trainer import ModelFactory


class LassoAnalyzer:
    """LASSO超参数分析器"""

    def __init__(
        self,
        alphas: List[float] = None,
        n_folds: int = 4,
        random_state: int = 42
    ):
        """
        初始化LASSO分析器

        Parameters
        ----------
        alphas : List[float], optional
            要测试的alpha值列表
        n_folds : int
            交叉验证折数
        random_state : int
            随机种子
        """
        if alphas is None:
            # 使用更密集的网格搜索
            alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 5.0, 10.0]

        self.alphas = alphas
        self.n_folds = n_folds
        self.random_state = random_state
        self.model_factory = ModelFactory(random_state=random_state)
        self.cv = CrossValidator(n_folds=n_folds, random_state=random_state)
        self.cv_results: List[CrossValidationResult] = []

    def run_grid_search(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        verbose: bool = True
    ) -> List[CrossValidationResult]:
        """
        对所有alpha值运行交叉验证

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            特征矩阵
        y : np.ndarray | pd.Series
            目标变量
        verbose : bool
            是否打印详细信息

        Returns
        -------
        List[CrossValidationResult]
            所有alpha值的交叉验证结果
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"LASSO 超参数网格搜索 ({len(self.alphas)} 个alpha值)")
            print("=" * 70)

        self.cv_results = []

        for idx, alpha in enumerate(self.alphas, 1):
            if verbose:
                print(f"\n[{idx}/{len(self.alphas)}] 测试 alpha = {alpha:.2e}")

            # 创建LASSO模型
            model = self.model_factory.get_lasso(alpha=alpha)

            # 运行交叉验证
            cv_result = self.cv.run_cv(
                model=model,
                X=X,
                y=y,
                model_name=f"Lasso_alpha_{alpha:.2e}",
                verbose=False
            )

            self.cv_results.append(cv_result)

            if verbose:
                # 提取关键指标
                ic_mean = cv_result.aggregate_metrics["val_ic_pearson_mean"]
                ic_std = cv_result.aggregate_metrics["val_ic_pearson_std"]
                rmse_mean = cv_result.aggregate_metrics["val_rmse_mean"]
                print(f"  验证集 IC: {ic_mean:.6f} ± {ic_std:.6f}")
                print(f"  验证集 RMSE: {rmse_mean:.6f}")

        if verbose:
            print("\n" + "=" * 70)
            print("网格搜索完成！")
            print("=" * 70)

        return self.cv_results

    def create_ic_fold_matrix(self, metric: str = "ic_pearson") -> pd.DataFrame:
        """
        创建 alpha × fold 的IC矩阵

        Parameters
        ----------
        metric : str
            指标名称，可选 "ic_pearson" 或 "ic_spearman"

        Returns
        -------
        pd.DataFrame
            行为alpha值，列为fold，值为IC
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        rows = []
        for cv_result in self.cv_results:
            # 提取alpha值
            alpha_str = cv_result.model_name.split("_")[-1]
            alpha = float(alpha_str)

            # 提取每个fold的IC
            row = {"alpha": alpha}
            for fold_result in cv_result.fold_results:
                fold_idx = fold_result.fold_idx
                ic_value = fold_result.val_metrics[metric]
                row[f"fold_{fold_idx}"] = ic_value

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("alpha")
        df = df.set_index("alpha")

        return df

    def create_quantile_ic_matrix(
        self,
        quantile_label: str = "top_10%_ic_pearson"
    ) -> pd.DataFrame:
        """
        创建 alpha × fold 的分位数IC矩阵

        Parameters
        ----------
        quantile_label : str
            分位数IC指标名称，例如 "top_10%_ic_pearson", "bottom_10%_ic_pearson"

        Returns
        -------
        pd.DataFrame
            行为alpha值，列为fold，值为分位数IC
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        rows = []
        for cv_result in self.cv_results:
            alpha_str = cv_result.model_name.split("_")[-1]
            alpha = float(alpha_str)

            row = {"alpha": alpha}
            for fold_result in cv_result.fold_results:
                fold_idx = fold_result.fold_idx
                if quantile_label in fold_result.val_metrics:
                    ic_value = fold_result.val_metrics[quantile_label]
                    row[f"fold_{fold_idx}"] = ic_value
                else:
                    row[f"fold_{fold_idx}"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("alpha")
        df = df.set_index("alpha")

        return df

    def compute_stability_metrics(self) -> pd.DataFrame:
        """
        计算每个alpha的稳定性指标

        Returns
        -------
        pd.DataFrame
            包含均值、标准差、变异系数等稳定性指标的表格
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        rows = []
        for cv_result in self.cv_results:
            alpha_str = cv_result.model_name.split("_")[-1]
            alpha = float(alpha_str)

            # 提取所有fold的IC
            ic_values = [
                fr.val_metrics["ic_pearson"]
                for fr in cv_result.fold_results
            ]

            row = {
                "alpha": alpha,
                "ic_mean": np.mean(ic_values),
                "ic_std": np.std(ic_values),
                "ic_cv": np.std(ic_values) / abs(np.mean(ic_values)) if np.mean(ic_values) != 0 else np.inf,
                "ic_min": np.min(ic_values),
                "ic_max": np.max(ic_values),
                "ic_range": np.max(ic_values) - np.min(ic_values),
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("alpha")

        return df

    def export_results(self, output_dir: str | Path = "results/lasso_analysis") -> None:
        """
        导出所有分析结果

        Parameters
        ----------
        output_dir : str | Path
            输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n导出LASSO分析结果至: {output_dir}")

        # 1. 导出IC fold矩阵
        ic_pearson_matrix = self.create_ic_fold_matrix("ic_pearson")
        ic_pearson_matrix.to_csv(output_dir / "lasso_ic_pearson_fold_matrix.csv")
        print(f"  ✓ IC (Pearson) fold矩阵")

        ic_spearman_matrix = self.create_ic_fold_matrix("ic_spearman")
        ic_spearman_matrix.to_csv(output_dir / "lasso_ic_spearman_fold_matrix.csv")
        print(f"  ✓ IC (Spearman) fold矩阵")

        # 2. 导出分位数IC矩阵
        quantile_labels = [
            "top_1%_ic_pearson", "top_5%_ic_pearson", "top_10%_ic_pearson",
            "bottom_1%_ic_pearson", "bottom_5%_ic_pearson", "bottom_10%_ic_pearson"
        ]

        for label in quantile_labels:
            try:
                matrix = self.create_quantile_ic_matrix(label)
                filename = f"lasso_{label.replace('%', 'pct')}_fold_matrix.csv"
                matrix.to_csv(output_dir / filename)
                print(f"  ✓ {label} fold矩阵")
            except Exception as e:
                print(f"  ⚠ 跳过 {label}: {e}")

        # 3. 导出稳定性指标
        stability = self.compute_stability_metrics()
        stability.to_csv(output_dir / "lasso_stability_metrics.csv", index=False)
        print(f"  ✓ 稳定性指标")

        # 4. 导出详细的fold结果
        all_fold_details = []
        for cv_result in self.cv_results:
            df_fold = self.cv.export_fold_details(cv_result)
            all_fold_details.append(df_fold)

        df_all = pd.concat(all_fold_details, ignore_index=True)
        df_all.to_csv(output_dir / "lasso_all_fold_details.csv", index=False)
        print(f"  ✓ 所有fold详细结果")

        print("\n所有结果导出完成！")

    def get_best_alpha(self, metric: str = "val_ic_pearson_mean") -> tuple[float, float]:
        """
        获取最佳alpha值

        Parameters
        ----------
        metric : str
            用于选择最佳alpha的指标

        Returns
        -------
        tuple[float, float]
            (最佳alpha值, 对应的指标值)
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        best_alpha = None
        best_score = -np.inf

        for cv_result in self.cv_results:
            alpha_str = cv_result.model_name.split("_")[-1]
            alpha = float(alpha_str)
            score = cv_result.aggregate_metrics[metric]

            if score > best_score:
                best_score = score
                best_alpha = alpha

        return best_alpha, best_score


if __name__ == "__main__":
    """
    直接运行此脚本时，使用真实数据进行LASSO超参数分析
    """
    from src.s01_data_analysis.data_loader import DataLoader

    print("\n" + "=" * 80)
    print("LASSO超参数分析 - 独立运行模式")
    print("=" * 80)

    # 加载真实数据
    loader = DataLoader(
        data_path="data/data.csv",
        target_column="realY",
        test_size=0.2,
        random_state=42
    )
    X_insample, X_outsample, y_insample, y_outsample = loader.load_and_split()

    print(f"\n使用真实数据:")
    print(f"  样本内数据: {X_insample.shape}")
    print(f"  特征数: {X_insample.shape[1]}")

    # 创建LASSO分析器（使用完整的alpha网格）
    analyzer = LassoAnalyzer(
        alphas=None,  # 使用默认的13个alpha值
        n_folds=4,
        random_state=42
    )

    # 运行网格搜索
    analyzer.run_grid_search(X_insample, y_insample, verbose=True)

    # 获取IC矩阵
    ic_matrix = analyzer.create_ic_fold_matrix()
    print("\n" + "=" * 80)
    print("IC (Pearson) 矩阵 (alpha × fold):")
    print("=" * 80)
    print(ic_matrix)

    # 获取稳定性指标
    stability = analyzer.compute_stability_metrics()
    print("\n" + "=" * 80)
    print("稳定性指标:")
    print("=" * 80)
    print(stability)

    # 获取最佳alpha
    best_alpha, best_score = analyzer.get_best_alpha()
    print("\n" + "=" * 80)
    print(f"最佳 alpha: {best_alpha:.2e} (IC = {best_score:.6f})")
    print("=" * 80)

    # 导出结果
    analyzer.export_results("results/lasso_analysis")
    print("\n分析完成！结果已保存至 results/lasso_analysis/")
