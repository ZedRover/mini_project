#!/usr/bin/env python3
"""
LightGBM模型训练和特征筛选
使用feature importance进行特征重要性排序
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.s02_model_training.cross_validation import CrossValidator
from src.s01_data_analysis.data_loader import DataLoader
from src.s02_model_training.metrics import MetricsCalculator


class LightGBMFeatureSelector:
    """LightGBM模型训练和特征筛选器"""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_folds: int = 4,
        random_state: int = 42,
        output_dir: str = "results/feature_selection/lightgbm"
    ):
        """
        初始化LightGBM特征筛选器

        Parameters
        ----------
        n_estimators : int
            树的数量
        learning_rate : float
            学习率
        num_leaves : int
            叶子节点数
        n_folds : int
            交叉验证折数
        random_state : int
            随机种子
        output_dir : str
            输出目录
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
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
        importance_type: str = 'gain'
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        训练LightGBM模型并筛选特征

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        top_k : int
            选择前k个重要特征
        importance_type : str
            重要性类型: 'gain', 'split' 或 'both'

        Returns
        -------
        selected_features : List[str]
            筛选出的特征列表
        feature_importance : pd.DataFrame
            特征重要性表格
        """
        print("\n" + "="*80)
        print("LightGBM模型训练和特征筛选")
        print("="*80)

        self.feature_names = list(X.columns)

        # 1. 基线模型（使用所有特征）
        print(f"\n[1/4] 训练基线LightGBM模型 (所有{len(self.feature_names)}个特征)")
        baseline_model = self._create_lgbm_model()
        baseline_cv_result = self.cv.run_cv(
            baseline_model, X, y,
            model_name="LightGBM_baseline",
            verbose=False
        )
        self.baseline_metrics = baseline_cv_result.aggregate_metrics

        print(f"  ✓ 基线模型 IC: {self.baseline_metrics['val_ic_pearson_mean']:.6f} ± {self.baseline_metrics['val_ic_pearson_std']:.6f}")
        print(f"  ✓ 基线模型 RMSE: {self.baseline_metrics['val_rmse_mean']:.6f}")

        # 2. 在全部数据上训练获取特征重要性
        print(f"\n[2/4] 在全部样本内数据上训练获取特征重要性")
        full_model = self._create_lgbm_model()
        full_model.fit(X, y)

        # 3. 提取特征重要性
        print(f"\n[3/4] 提取特征重要性")

        importance_dict = {
            'feature': self.feature_names,
            'importance_gain': full_model.feature_importances_,  # 默认是gain
        }

        # 如果需要，也获取split importance
        if importance_type in ['split', 'both']:
            full_model_split = self._create_lgbm_model()
            full_model_split.set_params(importance_type='split')
            full_model_split.fit(X, y)
            importance_dict['importance_split'] = full_model_split.feature_importances_

        self.feature_importance = pd.DataFrame(importance_dict)

        # 根据gain排序
        self.feature_importance = self.feature_importance.sort_values(
            'importance_gain', ascending=False
        ).reset_index(drop=True)

        # 统计有重要性的特征
        n_important = (self.feature_importance['importance_gain'] > 0).sum()
        print(f"  ✓ 有重要性（gain>0）的特征数: {n_important}/{len(self.feature_names)}")
        print(f"  ✓ 重要性范围: [{self.feature_importance['importance_gain'].min():.2f}, {self.feature_importance['importance_gain'].max():.2f}]")

        # 4. 特征筛选
        print(f"\n[4/4] 特征筛选 (Top-{top_k})")

        self.selected_features = self.feature_importance.head(top_k)['feature'].tolist()

        print(f"  ✓ 选择特征数: {len(self.selected_features)}")
        print(f"  ✓ 选择特征的重要性占比: {self.feature_importance.head(top_k)['importance_gain'].sum() / self.feature_importance['importance_gain'].sum():.2%}")

        # 展示前20个重要特征
        print(f"\n  前20个最重要特征:")
        for i, row in self.feature_importance.head(20).iterrows():
            importance_str = f"gain: {row['importance_gain']:8.2f}"
            if 'importance_split' in row:
                importance_str += f"  split: {row['importance_split']:6.0f}"
            print(f"    {i+1:2d}. {row['feature']:8s}  {importance_str}")

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

        selected_model = self._create_lgbm_model()
        selected_cv_result = self.cv.run_cv(
            selected_model, X_selected, y,
            model_name=f"LightGBM_selected_{len(self.selected_features)}features",
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
        print("导出LightGBM特征筛选结果")
        print(f"{'='*80}")

        # 1. 特征重要性
        importance_path = self.output_dir / "lightgbm_feature_importance.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"  ✓ 特征重要性: {importance_path}")

        # 2. 筛选的特征列表
        selected_path = self.output_dir / "lightgbm_selected_features.json"
        with open(selected_path, 'w') as f:
            json.dump({
                'n_features': len(self.selected_features),
                'features': self.selected_features
            }, f, indent=2)
        print(f"  ✓ 筛选特征列表: {selected_path}")

        # 3. 性能对比
        comparison_path = self.output_dir / "lightgbm_performance_comparison.csv"
        comparison_df = pd.DataFrame({
            'model': ['baseline_all_features', f'selected_{len(self.selected_features)}_features'],
            'n_features': [len(self.feature_names), len(self.selected_features)],
            **{k: [self.baseline_metrics.get(k, np.nan), self.selected_metrics.get(k, np.nan)]
               for k in self.baseline_metrics.keys()}
        })
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✓ 性能对比: {comparison_path}")

        print(f"\n所有结果已保存至: {self.output_dir}")

    def _create_lgbm_model(self) -> LGBMRegressor:
        """创建LightGBM模型"""
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True
        )


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

    # LightGBM特征筛选
    lgbm_selector = LightGBMFeatureSelector(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_folds=4,
        random_state=42,
        output_dir="results/feature_selection/lightgbm"
    )

    # 训练和筛选
    selected_features, feature_importance = lgbm_selector.train_and_select_features(
        X_insample,
        y_insample,
        top_k=100,
        importance_type='gain'
    )

    # 评估筛选后的特征
    lgbm_selector.evaluate_selected_features(X_insample, y_insample)

    # 导出结果
    lgbm_selector.export_results()

    print(f"\n{'='*80}")
    print("LightGBM特征筛选完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
