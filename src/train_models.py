#!/usr/bin/env python3
"""
主训练脚本
整合所有模块，运行完整的模型训练和分析流程
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from cross_validation import CrossValidator, compare_cv_results
from data_loader import DataLoader
from lasso_analysis import LassoAnalyzer
from metrics import MetricsCalculator, print_metrics_summary
from model_trainer import ModelFactory
from visualization import Visualizer

warnings.filterwarnings("ignore")


class ModelTrainingPipeline:
    """完整的模型训练流程"""

    def __init__(
        self,
        data_path: str = "data/data.csv",
        output_dir: str = "results",
        random_state: int = 42,
        n_folds: int = 4
    ):
        """
        初始化训练流程

        Parameters
        ----------
        data_path : str
            数据文件路径
        output_dir : str
            结果输出目录
        random_state : int
            随机种子
        n_folds : int
            交叉验证折数
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.n_folds = n_folds

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "lasso_analysis").mkdir(exist_ok=True)

        # 初始化各个组件
        self.data_loader = DataLoader(
            data_path=data_path,
            target_column="realY",
            test_size=0.2,
            random_state=random_state
        )
        self.model_factory = ModelFactory(random_state=random_state)
        self.cv = CrossValidator(n_folds=n_folds, random_state=random_state)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(output_dir=self.output_dir / "figures")

        # 存储结果
        self.X_insample = None
        self.X_outsample = None
        self.y_insample = None
        self.y_outsample = None
        self.baseline_cv_results = []
        self.lasso_analyzer = None

    def run(self, run_lasso_grid: bool = True, run_baseline_models: bool = True):
        """
        运行完整的训练流程

        Parameters
        ----------
        run_lasso_grid : bool
            是否运行LASSO网格搜索
        run_baseline_models : bool
            是否运行基线模型
        """
        print("\n" + "=" * 80)
        print("=" * 80)
        print("模型训练流程启动".center(80))
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("=" * 80)
        print("=" * 80)

        # 1. 加载和划分数据
        self._load_data()

        # 2. 运行基线模型
        if run_baseline_models:
            self._train_baseline_models()

        # 3. 运行LASSO网格搜索
        if run_lasso_grid:
            self._run_lasso_grid_search()

        # 4. 在样本外数据上评估最佳模型
        self._evaluate_on_outsample()

        # 5. 生成报告
        self._generate_report()

        print("\n" + "=" * 80)
        print("=" * 80)
        print("模型训练流程完成！".center(80))
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print(f"结果保存在: {self.output_dir}".center(80))
        print("=" * 80)
        print("=" * 80)

    def _load_data(self):
        """加载和划分数据"""
        self.X_insample, self.X_outsample, self.y_insample, self.y_outsample = (
            self.data_loader.load_and_split()
        )

    def _train_baseline_models(self):
        """训练基线模型"""
        print("\n" + "=" * 80)
        print("第一阶段: 基线模型训练（4-Fold交叉验证）")
        print("=" * 80)

        baseline_models = self.model_factory.get_all_baseline_models()

        for model_name, model in baseline_models.items():
            print(f"\n{'=' * 80}")
            print(f"训练模型: {model_name}")
            print("=" * 80)

            cv_result = self.cv.run_cv(
                model=model,
                X=self.X_insample,
                y=self.y_insample,
                model_name=model_name,
                verbose=True
            )

            self.baseline_cv_results.append(cv_result)

        # 导出基线模型结果
        print("\n" + "-" * 80)
        print("导出基线模型结果...")
        print("-" * 80)

        # 保存对比结果
        comparison_df = compare_cv_results(self.baseline_cv_results)
        comparison_df.to_csv(
            self.output_dir / "metrics" / "baseline_models_comparison.csv",
            index=False
        )
        print("  ✓ 基线模型对比表")

        # 保存详细fold结果
        for cv_result in self.baseline_cv_results:
            fold_df = self.cv.export_fold_details(cv_result)
            filename = f"{cv_result.model_name}_fold_details.csv"
            fold_df.to_csv(
                self.output_dir / "metrics" / filename,
                index=False
            )

        # 生成可视化
        print("\n生成基线模型可视化...")
        self.visualizer.plot_model_comparison(
            self.baseline_cv_results,
            filename="baseline_models_comparison.png"
        )
        self.visualizer.plot_ic_boxplot(
            self.baseline_cv_results,
            filename="baseline_ic_boxplot.png"
        )
        self.visualizer.plot_quantile_performance(
            self.baseline_cv_results,
            filename="baseline_quantile_performance.png"
        )

        print("\n" + "=" * 80)
        print("基线模型训练完成！")
        print("=" * 80)

    def _run_lasso_grid_search(self):
        """运行LASSO网格搜索"""
        print("\n" + "=" * 80)
        print("第二阶段: LASSO超参数网格搜索")
        print("=" * 80)

        # 定义更密集的alpha网格
        alphas = [
            1e-5, 5e-5, 1e-4, 5e-4,
            1e-3, 5e-3, 1e-2, 5e-2,
            1e-1, 0.5, 1.0, 5.0, 10.0
        ]

        self.lasso_analyzer = LassoAnalyzer(
            alphas=alphas,
            n_folds=self.n_folds,
            random_state=self.random_state
        )

        # 运行网格搜索
        lasso_cv_results = self.lasso_analyzer.run_grid_search(
            X=self.X_insample,
            y=self.y_insample,
            verbose=True
        )

        # 导出LASSO结果
        print("\n" + "-" * 80)
        print("导出LASSO分析结果...")
        print("-" * 80)
        self.lasso_analyzer.export_results(
            output_dir=self.output_dir / "lasso_analysis"
        )

        # 生成LASSO可视化
        print("\n生成LASSO可视化...")

        # IC热力图
        ic_pearson_matrix = self.lasso_analyzer.create_ic_fold_matrix("ic_pearson")
        self.visualizer.plot_lasso_ic_heatmap(
            ic_pearson_matrix,
            title="LASSO: Pearson IC across Alphas and Folds",
            filename="lasso_ic_pearson_heatmap.png"
        )

        ic_spearman_matrix = self.lasso_analyzer.create_ic_fold_matrix("ic_spearman")
        self.visualizer.plot_lasso_ic_heatmap(
            ic_spearman_matrix,
            title="LASSO: Spearman IC across Alphas and Folds",
            filename="lasso_ic_spearman_heatmap.png"
        )

        # 稳定性分析
        stability_df = self.lasso_analyzer.compute_stability_metrics()
        self.visualizer.plot_lasso_stability(
            stability_df,
            filename="lasso_stability_analysis.png"
        )

        # 分位数IC热力图
        for quantile_label in ["top_10%_ic_pearson", "bottom_10%_ic_pearson"]:
            try:
                q_matrix = self.lasso_analyzer.create_quantile_ic_matrix(quantile_label)
                self.visualizer.plot_lasso_ic_heatmap(
                    q_matrix,
                    title=f"LASSO: {quantile_label} across Alphas and Folds",
                    filename=f"lasso_{quantile_label.replace('%', 'pct')}_heatmap.png"
                )
            except Exception as e:
                print(f"  ⚠ 跳过 {quantile_label} 热力图: {e}")

        # 获取最佳alpha
        best_alpha, best_ic = self.lasso_analyzer.get_best_alpha()
        print(f"\n{'=' * 80}")
        print(f"最佳 Alpha: {best_alpha:.2e}")
        print(f"最佳验证集 IC: {best_ic:.6f}")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("LASSO网格搜索完成！")
        print("=" * 80)

    def _evaluate_on_outsample(self):
        """在样本外数据上评估最佳模型"""
        print("\n" + "=" * 80)
        print("第三阶段: 样本外数据评估")
        print("=" * 80)

        # 找出交叉验证表现最好的模型
        all_cv_results = self.baseline_cv_results.copy()
        if self.lasso_analyzer:
            all_cv_results.extend(self.lasso_analyzer.cv_results)

        # 按验证集IC排序
        sorted_results = sorted(
            all_cv_results,
            key=lambda x: x.aggregate_metrics["val_ic_pearson_mean"],
            reverse=True
        )

        # 取前5个模型
        top_k = min(5, len(sorted_results))
        print(f"\n选取验证集表现最好的 {top_k} 个模型在样本外数据上评估:\n")

        outsample_results = []

        for idx, cv_result in enumerate(sorted_results[:top_k], 1):
            model_name = cv_result.model_name
            print(f"[{idx}/{top_k}] {model_name}")

            # 使用交叉验证的第一个fold模型（或重新训练）
            # 这里重新在全部样本内数据上训练
            if "Lasso_alpha_" in model_name and self.lasso_analyzer:
                # 提取alpha值
                alpha_str = model_name.split("_")[-1]
                alpha = float(alpha_str)
                model = self.model_factory.get_lasso(alpha=alpha)
            else:
                # 使用model_factory创建新模型
                if model_name == "LinearRegression":
                    model = self.model_factory.get_linear_regression()
                elif model_name == "Ridge":
                    model = self.model_factory.get_ridge()
                elif model_name == "Lasso":
                    model = self.model_factory.get_lasso()
                elif model_name == "LightGBM":
                    model = self.model_factory.get_lightgbm()
                elif model_name == "NeuralNetwork":
                    model = self.model_factory.get_neural_network()
                else:
                    continue

            # 在全部样本内数据上训练
            model.fit(self.X_insample, self.y_insample)

            # 在样本外数据上预测
            y_pred = model.predict(self.X_outsample)

            # 计算指标
            metrics = self.metrics_calculator.compute_all_metrics(
                self.y_outsample.values, y_pred, include_quantile=True
            )

            print_metrics_summary(metrics, model_name)

            outsample_results.append({
                "model": model_name,
                **metrics
            })

            # 生成预测散点图
            self.visualizer.plot_prediction_scatter(
                self.y_outsample.values,
                y_pred,
                model_name=model_name,
                filename=f"outsample_{model_name}_scatter.png"
            )

        # 保存样本外结果
        outsample_df = pd.DataFrame(outsample_results)
        outsample_df.to_csv(
            self.output_dir / "metrics" / "outsample_evaluation.csv",
            index=False
        )
        print(f"\n样本外评估结果已保存至: {self.output_dir / 'metrics' / 'outsample_evaluation.csv'}")

        print("\n" + "=" * 80)
        print("样本外评估完成！")
        print("=" * 80)

    def _generate_report(self):
        """生成最终报告"""
        print("\n" + "=" * 80)
        print("生成最终报告...")
        print("=" * 80)

        report_path = self.output_dir / "training_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("模型训练报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据路径: {self.data_path}\n")
            f.write(f"随机种子: {self.random_state}\n")
            f.write(f"交叉验证折数: {self.n_folds}\n\n")

            # 数据摘要
            f.write("=" * 80 + "\n")
            f.write("数据摘要\n")
            f.write("=" * 80 + "\n")
            summary = self.data_loader.get_summary()
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            # 基线模型结果
            if self.baseline_cv_results:
                f.write("=" * 80 + "\n")
                f.write("基线模型交叉验证结果\n")
                f.write("=" * 80 + "\n")
                comparison_df = compare_cv_results(self.baseline_cv_results)
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n")

            # LASSO最佳alpha
            if self.lasso_analyzer:
                best_alpha, best_ic = self.lasso_analyzer.get_best_alpha()
                f.write("=" * 80 + "\n")
                f.write("LASSO最佳超参数\n")
                f.write("=" * 80 + "\n")
                f.write(f"最佳 Alpha: {best_alpha:.2e}\n")
                f.write(f"最佳验证集 IC: {best_ic:.6f}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 80 + "\n")

        print(f"  ✓ 训练报告已保存至: {report_path}")
        print("\n" + "=" * 80)
        print("报告生成完成！")
        print("=" * 80)


def main():
    """主函数"""
    pipeline = ModelTrainingPipeline(
        data_path="data/data.csv",
        output_dir="results",
        random_state=42,
        n_folds=4
    )

    pipeline.run(
        run_lasso_grid=True,
        run_baseline_models=True
    )


if __name__ == "__main__":
    main()
