#!/usr/bin/env python3
"""
LASSO模型训练和特征筛选
使用L1正则化的系数大小进行特征重要性排序
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径，支持 uv run 和 python -m 两种运行方式
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.cross_validation import CrossValidator
from src.s01_data_analysis.data_loader import DataLoader
from src.utils.metrics import MetricsCalculator


def load_best_alpha_from_tuning(
    tuning_results_dir: str = "results/lasso_analysis"
) -> float:
    """
    从LASSO超参数调优结果中读取最佳alpha值

    Parameters
    ----------
    tuning_results_dir : str
        LASSO调优结果目录

    Returns
    -------
    float
        最佳alpha值
    """
    result_dir = Path(tuning_results_dir)

    # 尝试读取稳定性指标文件
    stability_file = result_dir / "lasso_stability_metrics.csv"
    if stability_file.exists():
        df = pd.read_csv(stability_file)
        # 按IC均值排序，选择最佳alpha
        df = df.sort_values('ic_mean', ascending=False)
        best_alpha = df.iloc[0]['alpha']
        print(f"  从 {stability_file} 读取最佳alpha: {best_alpha}")
        return best_alpha

    # 如果没有找到文件，返回默认值
    print(f"  警告：未找到调优结果，使用默认alpha: 0.001")
    return 0.001


class LassoFeatureSelector:
    """LASSO模型训练和特征筛选器"""

    def __init__(
        self,
        alpha: float | str = "auto",
        n_folds: int = 4,
        random_state: int = 42,
        output_dir: str = "results/feature_selection/lasso"
    ):
        """
        初始化LASSO特征筛选器

        Parameters
        ----------
        alpha : float | str
            L1正则化强度。
            - 如果为"auto"，从超参数调优结果中自动读取最佳alpha
            - 如果为float，使用指定的alpha值
        n_folds : int
            交叉验证折数
        random_state : int
            随机种子
        output_dir : str
            输出目录
        """
        # 处理alpha参数
        if alpha == "auto":
            print("\n[自动选择alpha] 从超参数调优结果中读取最佳alpha...")
            self.alpha = load_best_alpha_from_tuning()
        else:
            self.alpha = alpha
        self.n_folds = n_folds
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cv = CrossValidator(n_folds=n_folds, random_state=random_state)
        self.metrics_calculator = MetricsCalculator()

        # 存储结果
        self.feature_names: List[str] = []
        self.feature_importance: pd.DataFrame | None = None
        self.selected_features: List[str] = []
        self.baseline_metrics: Dict = {}
        self.selected_metrics: Dict = {}

    def train_and_select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 100,
        threshold: float = 0.0
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        训练LASSO模型并筛选特征

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        top_k : int
            选择前k个重要特征
        threshold : float
            系数阈值（绝对值大于此值的特征被保留）

        Returns
        -------
        selected_features : List[str]
            筛选出的特征列表
        feature_importance : pd.DataFrame
            特征重要性表格
        """
        print("\n" + "="*80)
        print("LASSO模型训练和特征筛选")
        print("="*80)

        self.feature_names = list(X.columns)

        # 1. 基线模型（使用所有特征）
        print(f"\n[1/4] 训练基线LASSO模型 (alpha={self.alpha}, 所有{len(self.feature_names)}个特征)")
        baseline_model = self._create_lasso_model()
        baseline_cv_result = self.cv.run_cv(
            baseline_model, X, y,
            model_name=f"LASSO_baseline",
            verbose=False
        )
        self.baseline_metrics = baseline_cv_result.aggregate_metrics

        print(f"  ✓ 基线模型 IC: {self.baseline_metrics['val_ic_pearson_mean']:.6f} ± {self.baseline_metrics['val_ic_pearson_std']:.6f}")
        print(f"  ✓ 基线模型 RMSE: {self.baseline_metrics['val_rmse_mean']:.6f}")

        # 2. 在全部数据上训练获取系数
        print(f"\n[2/4] 在全部样本内数据上训练获取特征系数")
        full_model = self._create_lasso_model()
        full_model.fit(X, y)

        # 提取系数（考虑StandardScaler）
        if hasattr(full_model, 'named_steps'):
            coefficients = full_model.named_steps['model'].coef_
        else:
            coefficients = full_model.coef_

        # 3. 计算特征重要性
        print(f"\n[3/4] 计算特征重要性（基于系数绝对值）")
        importance_dict = {
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }
        self.feature_importance = pd.DataFrame(importance_dict)
        self.feature_importance = self.feature_importance.sort_values(
            'abs_coefficient', ascending=False
        ).reset_index(drop=True)

        # 统计非零系数
        n_nonzero = (np.abs(coefficients) > 1e-6).sum()
        print(f"  ✓ 非零系数特征数: {n_nonzero}/{len(self.feature_names)}")
        print(f"  ✓ 系数绝对值范围: [{np.abs(coefficients).min():.6f}, {np.abs(coefficients).max():.6f}]")

        # 4. 特征筛选
        print(f"\n[4/4] 特征筛选")
        print(f"  策略1: 选择Top {top_k}个特征")
        print(f"  策略2: 选择|系数| > {threshold}的特征")

        # 方法1: Top-K
        top_k_features = self.feature_importance.head(top_k)['feature'].tolist()

        # 方法2: 阈值筛选
        threshold_features = self.feature_importance[
            self.feature_importance['abs_coefficient'] > threshold
        ]['feature'].tolist()

        # 取两者的并集
        self.selected_features = list(set(top_k_features) | set(threshold_features))
        self.selected_features = sorted(self.selected_features)

        print(f"\n  ✓ Top-{top_k}特征数: {len(top_k_features)}")
        print(f"  ✓ 阈值筛选特征数: {len(threshold_features)}")
        print(f"  ✓ 最终选择特征数: {len(self.selected_features)}")

        # 展示前20个重要特征
        print(f"\n  前20个最重要特征:")
        for i, row in self.feature_importance.head(20).iterrows():
            print(f"    {i+1:2d}. {row['feature']:8s}  系数: {row['coefficient']:8.4f}  |系数|: {row['abs_coefficient']:.4f}")

        return self.selected_features, self.feature_importance

    def evaluate_selected_features(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        使用筛选后的特征评估模型性能

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量

        Returns
        -------
        Dict
            筛选后模型的评估指标
        """
        if not self.selected_features:
            raise ValueError("请先运行 train_and_select_features()")

        print(f"\n[评估] 使用筛选后的{len(self.selected_features)}个特征训练模型")

        # 只使用筛选后的特征
        X_selected = X[self.selected_features]

        selected_model = self._create_lasso_model()
        selected_cv_result = self.cv.run_cv(
            selected_model, X_selected, y,
            model_name=f"LASSO_selected_{len(self.selected_features)}features",
            verbose=False
        )
        self.selected_metrics = selected_cv_result.aggregate_metrics

        print(f"  ✓ 筛选后模型 IC: {self.selected_metrics['val_ic_pearson_mean']:.6f} ± {self.selected_metrics['val_ic_pearson_std']:.6f}")
        print(f"  ✓ 筛选后模型 RMSE: {self.selected_metrics['val_rmse_mean']:.6f}")

        # 对比
        ic_change = self.selected_metrics['val_ic_pearson_mean'] - self.baseline_metrics['val_ic_pearson_mean']
        rmse_change = self.selected_metrics['val_rmse_mean'] - self.baseline_metrics['val_rmse_mean']

        print(f"\n  【性能变化】")
        print(f"    IC变化: {ic_change:+.6f} ({ic_change/self.baseline_metrics['val_ic_pearson_mean']*100:+.2f}%)")
        print(f"    RMSE变化: {rmse_change:+.6f} ({rmse_change/self.baseline_metrics['val_rmse_mean']*100:+.2f}%)")

        return self.selected_metrics

    def export_results(self):
        """导出结果"""
        print(f"\n{'='*80}")
        print("导出LASSO特征筛选结果")
        print(f"{'='*80}")

        # 1. 特征重要性
        importance_path = self.output_dir / "lasso_feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"  ✓ 特征重要性: {importance_path}")

        # 2. 筛选的特征列表
        selected_path = self.output_dir / "lasso_selected_features.json"
        with open(selected_path, 'w') as f:
            json.dump({
                'n_features': len(self.selected_features),
                'features': self.selected_features
            }, f, indent=2)
        print(f"  ✓ 筛选特征列表: {selected_path}")

        # 3. 性能对比
        comparison_path = self.output_dir / "lasso_performance_comparison.csv"
        comparison_df = pd.DataFrame({
            'model': ['baseline_all_features', f'selected_{len(self.selected_features)}_features'],
            'n_features': [len(self.feature_names), len(self.selected_features)],
            **{k: [self.baseline_metrics.get(k, np.nan), self.selected_metrics.get(k, np.nan)]
               for k in self.baseline_metrics.keys()}
        })
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✓ 性能对比: {comparison_path}")

        print(f"\n所有结果已保存至: {self.output_dir}")

    def _create_lasso_model(self) -> Pipeline:
        """创建LASSO模型"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(
                alpha=self.alpha,
                max_iter=10000,
                random_state=self.random_state,
                tol=1e-4
            ))
        ])


def main():
    """主函数"""
    # 加载数据
    loader = DataLoader(
        data_path="data/data.csv",
        target_column="realY",
        test_size=0.2,
        random_state=42
    )
    X_insample, X_outsample, y_insample, y_outsample = loader.load_and_split()

    # LASSO特征筛选
    lasso_selector = LassoFeatureSelector(
        alpha=0.001,
        n_folds=4,
        random_state=42,
        output_dir="results/feature_selection/lasso"
    )

    # 训练和筛选
    selected_features, feature_importance = lasso_selector.train_and_select_features(
        X_insample,
        y_insample,
        top_k=100,
        threshold=0.001
    )

    # 评估筛选后的特征
    lasso_selector.evaluate_selected_features(X_insample, y_insample)

    # 导出结果
    lasso_selector.export_results()

    print(f"\n{'='*80}")
    print("LASSO特征筛选完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
