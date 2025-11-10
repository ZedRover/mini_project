#!/usr/bin/env python3
"""
特征筛选完整实验流程
整合LASSO和LightGBM的特征筛选，并进行全面的性能对比分析
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.s01_data_analysis.data_loader import DataLoader
from src.s04_feature_selection.feature_selection_comparison import FeatureSelectionComparator
from src.s04_feature_selection.lasso_feature_selector import LassoFeatureSelector
from src.s04_feature_selection.lightgbm_feature_selector import LightGBMFeatureSelector
from src.s02_model_training.cross_validation import CrossValidator


class FeatureSelectionExperiment:
    """特征筛选完整实验"""

    def __init__(
        self,
        data_path: str = "data/data.csv",
        target_column: str = "realY",
        test_size: float = 0.2,
        n_folds: int = 4,
        random_state: int = 42,
        output_dir: str = "results/feature_selection"
    ):
        """
        初始化特征筛选实验

        Parameters
        ----------
        data_path : str
            数据文件路径
        target_column : str
            目标变量列名
        test_size : float
            样本外数据比例
        n_folds : int
            交叉验证折数
        random_state : int
            随机种子
        output_dir : str
            输出目录
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.n_folds = n_folds
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据加载器
        self.loader = DataLoader(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state
        )

        # 交叉验证器
        self.cv = CrossValidator(n_folds=n_folds, random_state=random_state)

        # 特征筛选器
        self.lasso_selector = None
        self.lgbm_selector = None
        self.comparator = None

        # 存储结果
        self.X_insample = None
        self.X_outsample = None
        self.y_insample = None
        self.y_outsample = None
        self.cross_model_results = {}

    def run_experiment(
        self,
        lasso_alpha: float = 0.001,
        lasso_top_k: int = 100,
        lgbm_top_k: int = 100,
        lgbm_n_estimators: int = 500
    ):
        """
        运行完整的特征筛选实验

        Parameters
        ----------
        lasso_alpha : float
            LASSO的正则化强度
        lasso_top_k : int
            LASSO选择的特征数量
        lgbm_top_k : int
            LightGBM选择的特征数量
        lgbm_n_estimators : int
            LightGBM的树数量
        """
        print("\n" + "="*80)
        print("特征筛选完整实验流程")
        print("="*80)

        # 1. 加载数据
        print("\n[步骤 1/7] 加载数据并划分样本")
        self.X_insample, self.X_outsample, self.y_insample, self.y_outsample = (
            self.loader.load_and_split()
        )
        print(f"  ✓ 样本内数据: {self.X_insample.shape}")
        print(f"  ✓ 样本外数据: {self.X_outsample.shape}")
        print(f"  ✓ 特征总数: {self.X_insample.shape[1]}")

        # 2. LASSO特征筛选
        print("\n[步骤 2/7] LASSO特征筛选")
        self.lasso_selector = LassoFeatureSelector(
            alpha=lasso_alpha,
            n_folds=self.n_folds,
            random_state=self.random_state,
            output_dir=str(self.output_dir / "lasso")
        )
        lasso_features, _ = self.lasso_selector.train_and_select_features(
            self.X_insample,
            self.y_insample,
            top_k=lasso_top_k,
            threshold=0.0
        )
        self.lasso_selector.evaluate_selected_features(self.X_insample, self.y_insample)
        self.lasso_selector.export_results()

        # 3. LightGBM特征筛选
        print("\n[步骤 3/7] LightGBM特征筛选")
        self.lgbm_selector = LightGBMFeatureSelector(
            n_estimators=lgbm_n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            n_folds=self.n_folds,
            random_state=self.random_state,
            output_dir=str(self.output_dir / "lightgbm")
        )
        lgbm_features, _ = self.lgbm_selector.train_and_select_features(
            self.X_insample,
            self.y_insample,
            top_k=lgbm_top_k,
            importance_type='gain'
        )
        self.lgbm_selector.evaluate_selected_features(self.X_insample, self.y_insample)
        self.lgbm_selector.export_results()

        # 4. 特征筛选对比分析
        print("\n[步骤 4/7] 特征筛选对比分析")
        self.comparator = FeatureSelectionComparator(
            output_dir=str(self.output_dir / "comparison")
        )
        self.comparator.load_feature_selection_results(
            lasso_dir=str(self.output_dir / "lasso"),
            lgbm_dir=str(self.output_dir / "lightgbm")
        )
        overlap_stats = self.comparator.analyze_feature_overlap()
        self.comparator.compare_feature_rankings()
        self.comparator.visualize_comparison(overlap_stats)
        self.comparator.generate_report(overlap_stats)

        print(f"\n  【特征重叠统计】")
        print(f"    LASSO选择特征数: {overlap_stats['lasso_count']}")
        print(f"    LightGBM选择特征数: {overlap_stats['lgbm_count']}")
        print(f"    交集特征数: {overlap_stats['intersection_count']}")
        print(f"    并集特征数: {overlap_stats['union_count']}")
        print(f"    Jaccard相似度: {overlap_stats['jaccard_similarity']:.4f}")

        # 5. 跨模型性能测试
        print("\n[步骤 5/7] 跨模型性能测试")
        self._run_cross_model_evaluation(lasso_features, lgbm_features)

        # 6. 生成综合性能对比报告
        print("\n[步骤 6/7] 生成综合性能对比报告")
        self._generate_comprehensive_report()

        # 7. 生成性能对比可视化
        print("\n[步骤 7/7] 生成性能对比可视化")
        self._visualize_performance_comparison()

        print("\n" + "="*80)
        print("特征筛选实验完成！")
        print(f"所有结果已保存至: {self.output_dir}")
        print("="*80 + "\n")

    def _run_cross_model_evaluation(
        self,
        lasso_features: list,
        lgbm_features: list
    ):
        """
        跨模型性能评估：
        1. LASSO模型使用LightGBM筛选的特征
        2. LightGBM模型使用LASSO筛选的特征
        3. 两个模型使用交集特征

        Parameters
        ----------
        lasso_features : list
            LASSO筛选的特征列表
        lgbm_features : list
            LightGBM筛选的特征列表
        """
        print("  测试不同特征集合对模型性能的影响...")

        # 交集特征
        intersection_features = list(set(lasso_features) & set(lgbm_features))

        # LASSO模型使用不同特征集
        print("\n  [5.1] LASSO模型性能测试")
        lasso_with_lgbm = self._evaluate_lasso_with_features(
            lgbm_features, "LASSO_with_LightGBM_features"
        )
        lasso_with_intersection = self._evaluate_lasso_with_features(
            intersection_features, "LASSO_with_intersection_features"
        )

        # LightGBM模型使用不同特征集
        print("\n  [5.2] LightGBM模型性能测试")
        lgbm_with_lasso = self._evaluate_lgbm_with_features(
            lasso_features, "LightGBM_with_LASSO_features"
        )
        lgbm_with_intersection = self._evaluate_lgbm_with_features(
            intersection_features, "LightGBM_with_intersection_features"
        )

        # 存储结果
        self.cross_model_results = {
            'lasso_with_lgbm_features': lasso_with_lgbm,
            'lasso_with_intersection': lasso_with_intersection,
            'lgbm_with_lasso_features': lgbm_with_lasso,
            'lgbm_with_intersection': lgbm_with_intersection
        }

    def _evaluate_lasso_with_features(
        self,
        features: list,
        model_name: str
    ) -> Dict:
        """使用指定特征评估LASSO模型"""
        if not features:
            print(f"    ⚠ {model_name}: 特征列表为空，跳过")
            return {}

        X_selected = self.X_insample[features]
        lasso_model = self.lasso_selector._create_lasso_model()

        cv_result = self.cv.run_cv(
            lasso_model, X_selected, self.y_insample,
            model_name=model_name,
            verbose=False
        )

        metrics = cv_result.aggregate_metrics
        print(f"    ✓ {model_name} (n={len(features)})")
        print(f"      IC: {metrics['val_ic_pearson_mean']:.6f} ± {metrics['val_ic_pearson_std']:.6f}")
        print(f"      RMSE: {metrics['val_rmse_mean']:.6f}")

        return {
            'n_features': len(features),
            'metrics': metrics
        }

    def _evaluate_lgbm_with_features(
        self,
        features: list,
        model_name: str
    ) -> Dict:
        """使用指定特征评估LightGBM模型"""
        if not features:
            print(f"    ⚠ {model_name}: 特征列表为空，跳过")
            return {}

        X_selected = self.X_insample[features]
        lgbm_model = self.lgbm_selector._create_lgbm_model()

        cv_result = self.cv.run_cv(
            lgbm_model, X_selected, self.y_insample,
            model_name=model_name,
            verbose=False
        )

        metrics = cv_result.aggregate_metrics
        print(f"    ✓ {model_name} (n={len(features)})")
        print(f"      IC: {metrics['val_ic_pearson_mean']:.6f} ± {metrics['val_ic_pearson_std']:.6f}")
        print(f"      RMSE: {metrics['val_rmse_mean']:.6f}")

        return {
            'n_features': len(features),
            'metrics': metrics
        }

    def _generate_comprehensive_report(self):
        """生成综合性能对比报告"""
        print("  生成综合性能对比报告...")

        # 收集所有模型的性能数据
        performance_data = []

        # LASSO模型
        if self.lasso_selector:
            # 基线
            performance_data.append({
                'model': 'LASSO',
                'feature_set': 'all_features',
                'n_features': len(self.lasso_selector.feature_names),
                'ic_mean': self.lasso_selector.baseline_metrics.get('val_ic_pearson_mean', np.nan),
                'ic_std': self.lasso_selector.baseline_metrics.get('val_ic_pearson_std', np.nan),
                'rmse_mean': self.lasso_selector.baseline_metrics.get('val_rmse_mean', np.nan),
                'rmse_std': self.lasso_selector.baseline_metrics.get('val_rmse_std', np.nan)
            })
            # 使用LASSO筛选的特征
            performance_data.append({
                'model': 'LASSO',
                'feature_set': 'lasso_selected',
                'n_features': len(self.lasso_selector.selected_features),
                'ic_mean': self.lasso_selector.selected_metrics.get('val_ic_pearson_mean', np.nan),
                'ic_std': self.lasso_selector.selected_metrics.get('val_ic_pearson_std', np.nan),
                'rmse_mean': self.lasso_selector.selected_metrics.get('val_rmse_mean', np.nan),
                'rmse_std': self.lasso_selector.selected_metrics.get('val_rmse_std', np.nan)
            })

        # LightGBM模型
        if self.lgbm_selector:
            # 基线
            performance_data.append({
                'model': 'LightGBM',
                'feature_set': 'all_features',
                'n_features': len(self.lgbm_selector.feature_names),
                'ic_mean': self.lgbm_selector.baseline_metrics.get('val_ic_pearson_mean', np.nan),
                'ic_std': self.lgbm_selector.baseline_metrics.get('val_ic_pearson_std', np.nan),
                'rmse_mean': self.lgbm_selector.baseline_metrics.get('val_rmse_mean', np.nan),
                'rmse_std': self.lgbm_selector.baseline_metrics.get('val_rmse_std', np.nan)
            })
            # 使用LightGBM筛选的特征
            performance_data.append({
                'model': 'LightGBM',
                'feature_set': 'lgbm_selected',
                'n_features': len(self.lgbm_selector.selected_features),
                'ic_mean': self.lgbm_selector.selected_metrics.get('val_ic_pearson_mean', np.nan),
                'ic_std': self.lgbm_selector.selected_metrics.get('val_ic_pearson_std', np.nan),
                'rmse_mean': self.lgbm_selector.selected_metrics.get('val_rmse_mean', np.nan),
                'rmse_std': self.lgbm_selector.selected_metrics.get('val_rmse_std', np.nan)
            })

        # 跨模型结果
        if 'lasso_with_lgbm_features' in self.cross_model_results:
            result = self.cross_model_results['lasso_with_lgbm_features']
            if result:
                performance_data.append({
                    'model': 'LASSO',
                    'feature_set': 'lgbm_selected',
                    'n_features': result['n_features'],
                    'ic_mean': result['metrics'].get('val_ic_pearson_mean', np.nan),
                    'ic_std': result['metrics'].get('val_ic_pearson_std', np.nan),
                    'rmse_mean': result['metrics'].get('val_rmse_mean', np.nan),
                    'rmse_std': result['metrics'].get('val_rmse_std', np.nan)
                })

        if 'lgbm_with_lasso_features' in self.cross_model_results:
            result = self.cross_model_results['lgbm_with_lasso_features']
            if result:
                performance_data.append({
                    'model': 'LightGBM',
                    'feature_set': 'lasso_selected',
                    'n_features': result['n_features'],
                    'ic_mean': result['metrics'].get('val_ic_pearson_mean', np.nan),
                    'ic_std': result['metrics'].get('val_ic_pearson_std', np.nan),
                    'rmse_mean': result['metrics'].get('val_rmse_mean', np.nan),
                    'rmse_std': result['metrics'].get('val_rmse_std', np.nan)
                })

        if 'lasso_with_intersection' in self.cross_model_results:
            result = self.cross_model_results['lasso_with_intersection']
            if result:
                performance_data.append({
                    'model': 'LASSO',
                    'feature_set': 'intersection',
                    'n_features': result['n_features'],
                    'ic_mean': result['metrics'].get('val_ic_pearson_mean', np.nan),
                    'ic_std': result['metrics'].get('val_ic_pearson_std', np.nan),
                    'rmse_mean': result['metrics'].get('val_rmse_mean', np.nan),
                    'rmse_std': result['metrics'].get('val_rmse_std', np.nan)
                })

        if 'lgbm_with_intersection' in self.cross_model_results:
            result = self.cross_model_results['lgbm_with_intersection']
            if result:
                performance_data.append({
                    'model': 'LightGBM',
                    'feature_set': 'intersection',
                    'n_features': result['n_features'],
                    'ic_mean': result['metrics'].get('val_ic_pearson_mean', np.nan),
                    'ic_std': result['metrics'].get('val_ic_pearson_std', np.nan),
                    'rmse_mean': result['metrics'].get('val_rmse_mean', np.nan),
                    'rmse_std': result['metrics'].get('val_rmse_std', np.nan)
                })

        # 转换为DataFrame
        performance_df = pd.DataFrame(performance_data)

        # 保存
        report_path = self.output_dir / "comprehensive_performance_report.csv"
        performance_df.to_csv(report_path, index=False)
        print(f"  ✓ 综合性能报告: {report_path}")

        return performance_df

    def _visualize_performance_comparison(self):
        """可视化性能对比"""
        print("  生成性能对比图表...")

        # 读取报告
        report_path = self.output_dir / "comprehensive_performance_report.csv"
        if not report_path.exists():
            print("  ⚠ 未找到综合性能报告，跳过可视化")
            return

        df = pd.read_csv(report_path)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Selection Performance Comparison', fontsize=16, fontweight='bold')

        # 1. IC对比（按特征集分组）
        ax1 = axes[0, 0]
        feature_sets = df['feature_set'].unique()
        x = np.arange(len(feature_sets))
        width = 0.35

        lasso_df = df[df['model'] == 'LASSO']
        lgbm_df = df[df['model'] == 'LightGBM']

        lasso_ic = [lasso_df[lasso_df['feature_set'] == fs]['ic_mean'].values[0]
                    if len(lasso_df[lasso_df['feature_set'] == fs]) > 0 else 0
                    for fs in feature_sets]
        lgbm_ic = [lgbm_df[lgbm_df['feature_set'] == fs]['ic_mean'].values[0]
                   if len(lgbm_df[lgbm_df['feature_set'] == fs]) > 0 else 0
                   for fs in feature_sets]

        ax1.bar(x - width/2, lasso_ic, width, label='LASSO', alpha=0.8)
        ax1.bar(x + width/2, lgbm_ic, width, label='LightGBM', alpha=0.8)
        ax1.set_xlabel('Feature Set')
        ax1.set_ylabel('IC (Pearson)')
        ax1.set_title('IC Performance by Feature Set')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_sets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. RMSE对比
        ax2 = axes[0, 1]
        lasso_rmse = [lasso_df[lasso_df['feature_set'] == fs]['rmse_mean'].values[0]
                      if len(lasso_df[lasso_df['feature_set'] == fs]) > 0 else 0
                      for fs in feature_sets]
        lgbm_rmse = [lgbm_df[lgbm_df['feature_set'] == fs]['rmse_mean'].values[0]
                     if len(lgbm_df[lgbm_df['feature_set'] == fs]) > 0 else 0
                     for fs in feature_sets]

        ax2.bar(x - width/2, lasso_rmse, width, label='LASSO', alpha=0.8)
        ax2.bar(x + width/2, lgbm_rmse, width, label='LightGBM', alpha=0.8)
        ax2.set_xlabel('Feature Set')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE by Feature Set')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_sets, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. 特征数量 vs IC
        ax3 = axes[1, 0]
        lasso_data = df[df['model'] == 'LASSO'].sort_values('n_features')
        lgbm_data = df[df['model'] == 'LightGBM'].sort_values('n_features')

        ax3.plot(lasso_data['n_features'], lasso_data['ic_mean'],
                'o-', label='LASSO', markersize=8, linewidth=2)
        ax3.fill_between(lasso_data['n_features'],
                         lasso_data['ic_mean'] - lasso_data['ic_std'],
                         lasso_data['ic_mean'] + lasso_data['ic_std'],
                         alpha=0.2)

        ax3.plot(lgbm_data['n_features'], lgbm_data['ic_mean'],
                's-', label='LightGBM', markersize=8, linewidth=2)
        ax3.fill_between(lgbm_data['n_features'],
                         lgbm_data['ic_mean'] - lgbm_data['ic_std'],
                         lgbm_data['ic_mean'] + lgbm_data['ic_std'],
                         alpha=0.2)

        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('IC (Pearson)')
        ax3.set_title('IC vs Number of Features')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. 性能变化表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        # 计算性能变化
        summary_data = []
        for model in ['LASSO', 'LightGBM']:
            model_df = df[df['model'] == model]
            baseline = model_df[model_df['feature_set'] == 'all_features']

            if len(baseline) > 0:
                baseline_ic = baseline['ic_mean'].values[0]
                baseline_rmse = baseline['rmse_mean'].values[0]

                for fs in model_df['feature_set'].unique():
                    if fs != 'all_features':
                        fs_data = model_df[model_df['feature_set'] == fs]
                        if len(fs_data) > 0:
                            ic_change = fs_data['ic_mean'].values[0] - baseline_ic
                            rmse_change = fs_data['rmse_mean'].values[0] - baseline_rmse
                            n_features = fs_data['n_features'].values[0]

                            summary_data.append([
                                f"{model}_{fs}",
                                f"{n_features}",
                                f"{ic_change:+.6f}",
                                f"{rmse_change:+.6f}"
                            ])

        if summary_data:
            table = ax4.table(cellText=summary_data,
                            colLabels=['Model_Features', 'N', 'ΔIC', 'ΔRMSE'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            ax4.set_title('Performance Changes vs Baseline', fontweight='bold', pad=20)

        plt.tight_layout()

        # 保存
        output_path = self.output_dir / "performance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 性能对比图表: {output_path}")


def main():
    """主函数"""
    # 创建实验
    experiment = FeatureSelectionExperiment(
        data_path="data/data.csv",
        target_column="realY",
        test_size=0.2,
        n_folds=4,
        random_state=42,
        output_dir="results/feature_selection"
    )

    # 运行完整实验
    experiment.run_experiment(
        lasso_alpha=0.001,      # LASSO正则化强度
        lasso_top_k=100,        # LASSO选择特征数
        lgbm_top_k=100,         # LightGBM选择特征数
        lgbm_n_estimators=500   # LightGBM树数量
    )


if __name__ == "__main__":
    main()
