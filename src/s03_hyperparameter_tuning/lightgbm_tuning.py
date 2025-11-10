#!/usr/bin/env python3
"""
LightGBM超参数分析模块
搜索最佳的n_estimators, learning_rate, num_leaves等参数
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

from src.utils.cross_validation import CrossValidationResult, CrossValidator
from src.s02_model_training.model_trainer import ModelFactory


class LightGBMTuner:
    """LightGBM超参数调优器"""

    def __init__(
        self,
        param_grid: Dict = None,
        n_folds: int = 4,
        random_state: int = 42
    ):
        """
        初始化LightGBM调优器

        Parameters
        ----------
        param_grid : Dict, optional
            参数网格。如果为None，使用默认网格
        n_folds : int
            交叉验证折数
        random_state : int
            随机种子
        """
        if param_grid is None:
            # 默认参数网格
            param_grid = {
                'n_estimators': [100, 300, 500, 700],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'max_depth': [-1],  # -1 表示无限制
                'min_child_samples': [20],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        self.param_grid = param_grid
        self.n_folds = n_folds
        self.random_state = random_state
        self.model_factory = ModelFactory(random_state=random_state)
        self.cv = CrossValidator(n_folds=n_folds, random_state=random_state)
        self.cv_results: List[CrossValidationResult] = []
        self.all_params: List[Dict] = []

    def _generate_param_combinations(self) -> List[Dict]:
        """生成所有参数组合"""
        from itertools import product

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def run_grid_search(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        verbose: bool = True,
        max_combinations: int = None
    ) -> List[CrossValidationResult]:
        """
        运行网格搜索

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            特征矩阵
        y : np.ndarray | pd.Series
            目标变量
        verbose : bool
            是否打印详细信息
        max_combinations : int, optional
            最多测试的参数组合数。如果为None，测试所有组合

        Returns
        -------
        List[CrossValidationResult]
            所有参数组合的交叉验证结果
        """
        # 生成所有参数组合
        param_combinations = self._generate_param_combinations()

        if max_combinations is not None:
            param_combinations = param_combinations[:max_combinations]

        if verbose:
            print("\n" + "=" * 80)
            print(f"LightGBM超参数网格搜索 ({len(param_combinations)} 个参数组合)")
            print("=" * 80)
            print(f"\n参数网格:")
            for key, values in self.param_grid.items():
                print(f"  {key}: {values}")

        self.cv_results = []
        self.all_params = []

        for idx, params in enumerate(param_combinations, 1):
            if verbose:
                print(f"\n[{idx}/{len(param_combinations)}] 测试参数组合:")
                for key, value in params.items():
                    print(f"  {key}: {value}")

            # 创建LightGBM模型
            model = self.model_factory.get_lightgbm(**params)

            # 构建模型名称
            model_name = f"LightGBM_" + "_".join([f"{k}={v}" for k, v in params.items()])

            # 运行交叉验证
            cv_result = self.cv.run_cv(
                model=model,
                X=X,
                y=y,
                model_name=model_name,
                verbose=False
            )

            self.cv_results.append(cv_result)
            self.all_params.append(params)

            if verbose:
                ic_mean = cv_result.aggregate_metrics["val_ic_pearson_mean"]
                ic_std = cv_result.aggregate_metrics["val_ic_pearson_std"]
                rmse_mean = cv_result.aggregate_metrics["val_rmse_mean"]
                print(f"  验证集 IC: {ic_mean:.6f} ± {ic_std:.6f}")
                print(f"  验证集 RMSE: {rmse_mean:.6f}")

        if verbose:
            print("\n" + "=" * 80)
            print("网格搜索完成！")
            print("=" * 80)

        return self.cv_results

    def get_best_params(self, metric: str = "val_ic_pearson_mean") -> tuple[Dict, float]:
        """
        获取最佳参数组合

        Parameters
        ----------
        metric : str
            用于选择最佳参数的指标

        Returns
        -------
        tuple[Dict, float]
            (最佳参数字典, 对应的指标值)
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        best_params = None
        best_score = -np.inf

        for params, cv_result in zip(self.all_params, self.cv_results):
            score = cv_result.aggregate_metrics[metric]

            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def create_results_dataframe(self) -> pd.DataFrame:
        """
        创建包含所有参数组合及其性能的DataFrame

        Returns
        -------
        pd.DataFrame
            包含参数和性能指标的表格
        """
        if not self.cv_results:
            raise ValueError("请先运行 run_grid_search()")

        rows = []
        for params, cv_result in zip(self.all_params, self.cv_results):
            row = {**params}  # 添加所有参数

            # 添加性能指标
            row.update({
                'ic_pearson_mean': cv_result.aggregate_metrics["val_ic_pearson_mean"],
                'ic_pearson_std': cv_result.aggregate_metrics["val_ic_pearson_std"],
                'ic_spearman_mean': cv_result.aggregate_metrics["val_ic_spearman_mean"],
                'rmse_mean': cv_result.aggregate_metrics["val_rmse_mean"],
                'rmse_std': cv_result.aggregate_metrics["val_rmse_std"],
                'r2_mean': cv_result.aggregate_metrics["val_r2_mean"],
            })

            rows.append(row)

        df = pd.DataFrame(rows)
        # 按IC排序
        df = df.sort_values('ic_pearson_mean', ascending=False)

        return df

    def export_results(self, output_dir: str | Path = "results/lightgbm_tuning") -> None:
        """
        导出所有分析结果

        Parameters
        ----------
        output_dir : str | Path
            输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n导出LightGBM调优结果至: {output_dir}")

        # 1. 导出所有参数组合的性能
        df_results = self.create_results_dataframe()
        df_results.to_csv(output_dir / "lightgbm_grid_search_results.csv", index=False)
        print(f"  ✓ 网格搜索结果: lightgbm_grid_search_results.csv")

        # 2. 导出最佳参数
        best_params, best_score = self.get_best_params()
        with open(output_dir / "best_params.txt", "w") as f:
            f.write("LightGBM最佳参数组合\n")
            f.write("=" * 50 + "\n\n")
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n最佳IC (Pearson): {best_score:.6f}\n")
        print(f"  ✓ 最佳参数: best_params.txt")

        # 3. 导出Top-10参数组合
        top10 = df_results.head(10)
        top10.to_csv(output_dir / "top10_param_combinations.csv", index=False)
        print(f"  ✓ Top-10参数组合: top10_param_combinations.csv")

        # 4. 导出详细的fold结果
        all_fold_details = []
        for cv_result in self.cv_results:
            df_fold = self.cv.export_fold_details(cv_result)
            all_fold_details.append(df_fold)

        df_all = pd.concat(all_fold_details, ignore_index=True)
        df_all.to_csv(output_dir / "lightgbm_all_fold_details.csv", index=False)
        print(f"  ✓ 所有fold详细结果: lightgbm_all_fold_details.csv")

        print("\n所有结果导出完成！")


if __name__ == "__main__":
    """
    直接运行此脚本时，使用真实数据进行LightGBM超参数搜索
    """
    from src.s01_data_analysis.data_loader import DataLoader

    print("\n" + "=" * 80)
    print("LightGBM超参数搜索 - 独立运行模式")
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

    # 创建LightGBM调优器（使用简化的参数网格用于快速测试）
    tuner = LightGBMTuner(
        param_grid={
            'n_estimators': [300, 500],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 63],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        },
        n_folds=4,
        random_state=42
    )

    # 运行网格搜索
    tuner.run_grid_search(X_insample, y_insample, verbose=True)

    # 获取最佳参数
    best_params, best_score = tuner.get_best_params()
    print("\n" + "=" * 80)
    print("最佳参数组合:")
    print("=" * 80)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n最佳IC (Pearson): {best_score:.6f}")
    print("=" * 80)

    # 导出结果
    tuner.export_results("results/lightgbm_tuning")
    print("\n分析完成！结果已保存至 results/lightgbm_tuning/")
