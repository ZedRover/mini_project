#!/usr/bin/env python3
"""
特征筛选结果对比分析
对比LASSO和LightGBM筛选出的特征，分析交集、并集、差异
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2

sns.set_style('whitegrid')


class FeatureSelectionComparator:
    """特征筛选对比分析器"""

    def __init__(self, output_dir: str = "results/feature_selection/comparison"):
        """
        初始化对比分析器

        Parameters
        ----------
        output_dir : str
            输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lasso_features: Set[str] = set()
        self.lgbm_features: Set[str] = set()
        self.lasso_importance: pd.DataFrame | None = None
        self.lgbm_importance: pd.DataFrame | None = None

    def load_feature_selection_results(
        self,
        lasso_dir: str = "results/feature_selection/lasso",
        lgbm_dir: str = "results/feature_selection/lightgbm"
    ):
        """
        加载两个模型的特征筛选结果

        Parameters
        ----------
        lasso_dir : str
            LASSO结果目录
        lgbm_dir : str
            LightGBM结果目录
        """
        print("\n" + "="*80)
        print("加载特征筛选结果")
        print("="*80)

        # 加载LASSO特征
        lasso_features_path = Path(lasso_dir) / "lasso_selected_features.json"
        with open(lasso_features_path, 'r') as f:
            lasso_data = json.load(f)
            self.lasso_features = set(lasso_data['features'])

        lasso_importance_path = Path(lasso_dir) / "lasso_feature_importance.csv"
        self.lasso_importance = pd.read_csv(lasso_importance_path)

        print(f"\n[LASSO]")
        print(f"  ✓ 筛选特征数: {len(self.lasso_features)}")

        # 加载LightGBM特征
        lgbm_features_path = Path(lgbm_dir) / "lightgbm_selected_features.json"
        with open(lgbm_features_path, 'r') as f:
            lgbm_data = json.load(f)
            self.lgbm_features = set(lgbm_data['features'])

        lgbm_importance_path = Path(lgbm_dir) / "lightgbm_feature_importance.csv"
        self.lgbm_importance = pd.read_csv(lgbm_importance_path)

        print(f"\n[LightGBM]")
        print(f"  ✓ 筛选特征数: {len(self.lgbm_features)}")

    def analyze_feature_overlap(self) -> Dict:
        """
        分析特征交集、并集、差异

        Returns
        -------
        Dict
            包含交集、并集、差异统计的字典
        """
        print("\n" + "="*80)
        print("特征集合分析")
        print("="*80)

        # 计算交集、并集、差异
        intersection = self.lasso_features & self.lgbm_features
        union = self.lasso_features | self.lgbm_features
        lasso_only = self.lasso_features - self.lgbm_features
        lgbm_only = self.lgbm_features - self.lasso_features

        # 统计
        stats = {
            'lasso_count': len(self.lasso_features),
            'lgbm_count': len(self.lgbm_features),
            'intersection_count': len(intersection),
            'union_count': len(union),
            'lasso_only_count': len(lasso_only),
            'lgbm_only_count': len(lgbm_only),
            'jaccard_similarity': len(intersection) / len(union) if len(union) > 0 else 0,
            'lasso_coverage': len(intersection) / len(self.lasso_features) if len(self.lasso_features) > 0 else 0,
            'lgbm_coverage': len(intersection) / len(self.lgbm_features) if len(self.lgbm_features) > 0 else 0
        }

        print(f"\n【集合统计】")
        print(f"  LASSO筛选特征数:     {stats['lasso_count']}")
        print(f"  LightGBM筛选特征数:  {stats['lgbm_count']}")
        print(f"  交集特征数:          {stats['intersection_count']}")
        print(f"  并集特征数:          {stats['union_count']}")
        print(f"  LASSO独有特征:       {stats['lasso_only_count']}")
        print(f"  LightGBM独有特征:    {stats['lgbm_only_count']}")

        print(f"\n【相似度指标】")
        print(f"  Jaccard相似度:       {stats['jaccard_similarity']:.4f}")
        print(f"  LASSO覆盖率:         {stats['lasso_coverage']:.4f} (交集/LASSO)")
        print(f"  LightGBM覆盖率:      {stats['lgbm_coverage']:.4f} (交集/LightGBM)")

        # 保存集合
        sets_dict = {
            'intersection': sorted(list(intersection)),
            'union': sorted(list(union)),
            'lasso_only': sorted(list(lasso_only)),
            'lgbm_only': sorted(list(lgbm_only))
        }

        sets_path = self.output_dir / "feature_sets.json"
        with open(sets_path, 'w') as f:
            json.dump(sets_dict, f, indent=2)
        print(f"\n  ✓ 特征集合已保存: {sets_path}")

        return stats

    def compare_feature_rankings(self, top_k: int = 50):
        """
        对比两个模型的特征排名

        Parameters
        ----------
        top_k : int
            对比前k个特征
        """
        print(f"\n{'='*80}")
        print(f"特征排名对比 (Top-{top_k})")
        print("="*80)

        # 获取Top-K特征
        lasso_top_k = self.lasso_importance.head(top_k)
        lgbm_top_k = self.lgbm_importance.head(top_k)

        # 创建排名字典
        lasso_ranks = {row['feature']: i+1 for i, row in lasso_top_k.iterrows()}
        lgbm_ranks = {row['feature']: i+1 for i, row in lgbm_top_k.iterrows()}

        # 找出在两个Top-K中都出现的特征
        common_features = set(lasso_ranks.keys()) & set(lgbm_ranks.keys())

        print(f"\n在两个模型Top-{top_k}中都出现的特征数: {len(common_features)}")

        # 对比排名
        rank_comparison = []
        for feature in common_features:
            rank_comparison.append({
                'feature': feature,
                'lasso_rank': lasso_ranks[feature],
                'lgbm_rank': lgbm_ranks[feature],
                'rank_diff': abs(lasso_ranks[feature] - lgbm_ranks[feature])
            })

        rank_df = pd.DataFrame(rank_comparison).sort_values('rank_diff')

        print(f"\n排名差异最小的10个特征:")
        for i, row in rank_df.head(10).iterrows():
            print(f"  {row['feature']:8s}  LASSO排名:{row['lasso_rank']:3.0f}  LGBM排名:{row['lgbm_rank']:3.0f}  差异:{row['rank_diff']:3.0f}")

        print(f"\n排名差异最大的10个特征:")
        for i, row in rank_df.tail(10).iterrows():
            print(f"  {row['feature']:8s}  LASSO排名:{row['lasso_rank']:3.0f}  LGBM排名:{row['lgbm_rank']:3.0f}  差异:{row['rank_diff']:3.0f}")

        # 保存排名对比
        rank_path = self.output_dir / "feature_rank_comparison.csv"
        rank_df.to_csv(rank_path, index=False)
        print(f"\n  ✓ 排名对比已保存: {rank_path}")

        return rank_df

    def visualize_comparison(self, stats: Dict):
        """
        可视化特征筛选对比

        Parameters
        ----------
        stats : Dict
            集合统计信息
        """
        print(f"\n{'='*80}")
        print("生成可视化对比图")
        print("="*80)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Venn图
        ax = fig.add_subplot(gs[0, 0])
        try:
            venn2(
                subsets=(
                    stats['lasso_only_count'],
                    stats['lgbm_only_count'],
                    stats['intersection_count']
                ),
                set_labels=('LASSO', 'LightGBM'),
                ax=ax
            )
            ax.set_title('Feature Selection Overlap', fontweight='bold', fontsize=12)
        except Exception as e:
            print(f"  ⚠ Venn图生成失败: {e}")
            ax.text(0.5, 0.5, 'Venn diagram unavailable',
                   ha='center', va='center', transform=ax.transAxes)

        # 2. 柱状图对比
        ax = fig.add_subplot(gs[0, 1])
        categories = ['LASSO\nSelected', 'LightGBM\nSelected', 'Intersection', 'Union']
        values = [
            stats['lasso_count'],
            stats['lgbm_count'],
            stats['intersection_count'],
            stats['union_count']
        ]
        bars = ax.bar(categories, values, color=['steelblue', 'coral', 'green', 'purple'], alpha=0.7)
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Count Comparison', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        # 3. 相似度指标
        ax = fig.add_subplot(gs[0, 2])
        metrics = ['Jaccard\nSimilarity', 'LASSO\nCoverage', 'LightGBM\nCoverage']
        metric_values = [
            stats['jaccard_similarity'],
            stats['lasso_coverage'],
            stats['lgbm_coverage']
        ]
        bars = ax.bar(metrics, metric_values, color=['purple', 'steelblue', 'coral'], alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.0])
        ax.set_title('Similarity Metrics', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4. Top-20特征对比（LASSO）
        ax = fig.add_subplot(gs[1, 0])
        lasso_top20 = self.lasso_importance.head(20)
        y_pos = np.arange(len(lasso_top20))
        ax.barh(y_pos, lasso_top20['abs_coefficient'].values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lasso_top20['feature'].values, fontsize=8)
        ax.set_xlabel('|Coefficient|')
        ax.set_title('LASSO Top-20 Features', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 5. Top-20特征对比（LightGBM）
        ax = fig.add_subplot(gs[1, 1])
        lgbm_top20 = self.lgbm_importance.head(20)
        y_pos = np.arange(len(lgbm_top20))
        ax.barh(y_pos, lgbm_top20['importance_gain'].values, color='coral', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lgbm_top20['feature'].values, fontsize=8)
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title('LightGBM Top-20 Features', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 6. 集合关系饼图
        ax = fig.add_subplot(gs[1, 2])
        sizes = [
            stats['intersection_count'],
            stats['lasso_only_count'],
            stats['lgbm_only_count']
        ]
        labels = [
            f"Both\n({stats['intersection_count']})",
            f"LASSO only\n({stats['lasso_only_count']})",
            f"LightGBM only\n({stats['lgbm_only_count']})"
        ]
        colors = ['green', 'steelblue', 'coral']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Feature Distribution', fontweight='bold', fontsize=12)

        plt.suptitle('Feature Selection Comparison: LASSO vs LightGBM',
                    fontsize=14, fontweight='bold', y=0.98)

        save_path = self.output_dir / "feature_selection_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path.name}")
        plt.close()

    def generate_report(self, stats: Dict):
        """
        生成对比报告

        Parameters
        ----------
        stats : Dict
            统计信息
        """
        report_path = self.output_dir / "comparison_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 特征筛选对比报告\n\n")
            f.write("## 1. 集合统计\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| LASSO筛选特征数 | {stats['lasso_count']} |\n")
            f.write(f"| LightGBM筛选特征数 | {stats['lgbm_count']} |\n")
            f.write(f"| 交集特征数 | {stats['intersection_count']} |\n")
            f.write(f"| 并集特征数 | {stats['union_count']} |\n")
            f.write(f"| LASSO独有特征 | {stats['lasso_only_count']} |\n")
            f.write(f"| LightGBM独有特征 | {stats['lgbm_only_count']} |\n\n")

            f.write("## 2. 相似度指标\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| Jaccard相似度 | {stats['jaccard_similarity']:.4f} |\n")
            f.write(f"| LASSO覆盖率 | {stats['lasso_coverage']:.4f} |\n")
            f.write(f"| LightGBM覆盖率 | {stats['lgbm_coverage']:.4f} |\n\n")

            f.write("## 3. 可视化\n\n")
            f.write("![对比图](feature_selection_comparison.png)\n\n")

            f.write("## 4. 结论\n\n")
            if stats['jaccard_similarity'] > 0.7:
                f.write("两个模型筛选的特征高度一致，说明这些特征具有稳健的预测能力。\n")
            elif stats['jaccard_similarity'] > 0.4:
                f.write("两个模型筛选的特征有一定重叠，但也各有侧重。\n")
            else:
                f.write("两个模型筛选的特征差异较大，可能因为模型机制不同导致关注点不同。\n")

        print(f"  ✓ 报告已生成: {report_path}")


def main():
    """主函数"""
    comparator = FeatureSelectionComparator(
        output_dir="results/feature_selection/comparison"
    )

    # 加载结果
    comparator.load_feature_selection_results()

    # 分析特征重叠
    stats = comparator.analyze_feature_overlap()

    # 对比排名
    comparator.compare_feature_rankings(top_k=50)

    # 可视化
    comparator.visualize_comparison(stats)

    # 生成报告
    comparator.generate_report(stats)

    print(f"\n{'='*80}")
    print("特征筛选对比分析完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
