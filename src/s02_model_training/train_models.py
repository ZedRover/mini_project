#!/usr/bin/env python3
"""
基线模型训练脚本
使用全量特征训练所有基线模型（不含超参数搜索）
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径，支持 uv run 和 python -m 两种运行方式
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.s02_model_training.cross_validation import CrossValidator, compare_cv_results
from src.s01_data_analysis.data_loader import DataLoader
from src.s02_model_training.metrics import MetricsCalculator, print_metrics_summary
from src.s02_model_training.model_trainer import ModelFactory

warnings.filterwarnings("ignore")


class BaselineModelTrainer:
    """基线模型训练器 - 使用全量特征和固定参数"""

    def __init__(
        self,
        data_path: str = "data/data.csv",
        output_dir: str = "results/baseline_models",
        random_state: int = 42,
        n_folds: int = 4
    ):
        """
        初始化训练器

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

        # 存储结果
        self.X_insample = None
        self.X_outsample = None
        self.y_insample = None
        self.y_outsample = None
        self.cv_results = []

    def run(self):
        """运行完整的基线模型训练流程"""
        print("\n" + "=" * 80)
        print("基线模型训练流程".center(80))
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("=" * 80)

        # 步骤1: 加载数据
        self._load_data()

        # 步骤2: 训练基线模型
        self._train_baseline_models()

        # 步骤3: 样本外评估
        self._evaluate_on_outsample()

        # 步骤4: 导出结果
        self._export_results()

        print("\n" + "=" * 80)
        print("训练流程完成！".center(80))
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print(f"所有结果已保存至: {self.output_dir}".center(80))
        print("=" * 80 + "\n")

    def _load_data(self):
        """加载和划分数据"""
        print("\n" + "=" * 80)
        print("[步骤 1/4] 加载数据")
        print("=" * 80)

        self.X_insample, self.X_outsample, self.y_insample, self.y_outsample = (
            self.data_loader.load_and_split()
        )

        print(f"\n数据划分完成:")
        print(f"  样本内数据: {self.X_insample.shape[0]} 样本 × {self.X_insample.shape[1]} 特征")
        print(f"  样本外数据: {self.X_outsample.shape[0]} 样本 × {self.X_outsample.shape[1]} 特征")

    def _train_baseline_models(self):
        """训练所有基线模型（使用全量特征）"""
        print("\n" + "=" * 80)
        print("[步骤 2/4] 训练基线模型（全量特征）")
        print("=" * 80)

        # 定义基线模型配置
        baseline_models = {
            "LinearRegression": self.model_factory.get_linear_regression(),
            "Ridge": self.model_factory.get_ridge(alpha=1.0),
            "Lasso": self.model_factory.get_lasso(alpha=0.01),  # 固定alpha
            "LightGBM": self.model_factory.get_lightgbm(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31
            )
        }

        print(f"\n将训练 {len(baseline_models)} 个基线模型:")
        for model_name in baseline_models.keys():
            print(f"  - {model_name}")
        print(f"\n使用 {self.n_folds}-fold 交叉验证")

        # 训练每个模型
        self.cv_results = []
        for idx, (model_name, model) in enumerate(baseline_models.items(), 1):
            print("\n" + "-" * 80)
            print(f"[{idx}/{len(baseline_models)}] 训练模型: {model_name}")
            print("-" * 80)

            # 运行交叉验证
            cv_result = self.cv.run_cv(
                model=model,
                X=self.X_insample,
                y=self.y_insample,
                model_name=model_name,
                verbose=True
            )

            self.cv_results.append(cv_result)

            # 打印关键指标
            ic_mean = cv_result.aggregate_metrics["val_ic_pearson_mean"]
            ic_std = cv_result.aggregate_metrics["val_ic_pearson_std"]
            rmse_mean = cv_result.aggregate_metrics["val_rmse_mean"]
            print(f"\n  交叉验证结果:")
            print(f"    IC (Pearson): {ic_mean:.6f} ± {ic_std:.6f}")
            print(f"    RMSE: {rmse_mean:.6f}")

        # 打印对比总结
        print("\n" + "=" * 80)
        print("模型对比总结")
        print("=" * 80)
        compare_cv_results(self.cv_results)

    def _evaluate_on_outsample(self):
        """在样本外数据上评估所有模型"""
        print("\n" + "=" * 80)
        print("[步骤 3/4] 样本外评估")
        print("=" * 80)

        outsample_results = []

        for cv_result in self.cv_results:
            model_name = cv_result.model_name
            print(f"\n评估模型: {model_name}")

            # 重新训练模型使用全部样本内数据
            model = cv_result.model
            model.fit(self.X_insample, self.y_insample)

            # 在样本外数据上预测
            y_pred = model.predict(self.X_outsample)

            # 计算指标
            metrics = self.metrics_calculator.compute_all_metrics(
                self.y_outsample, y_pred
            )

            outsample_results.append({
                "model": model_name,
                **metrics
            })

            # 打印关键指标
            print(f"  IC (Pearson): {metrics['ic_pearson']:.6f}")
            print(f"  IC (Spearman): {metrics['ic_spearman']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  R²: {metrics['r2']:.6f}")

        # 保存样本外结果
        self.outsample_df = pd.DataFrame(outsample_results)

        # 打印总结
        print("\n" + "=" * 80)
        print("样本外性能总结（按IC排序）")
        print("=" * 80)
        df_sorted = self.outsample_df.sort_values("ic_pearson", ascending=False)
        print(df_sorted[["model", "ic_pearson", "ic_spearman", "rmse", "r2"]].to_string(index=False))

    def _export_results(self):
        """导出所有结果"""
        print("\n" + "=" * 80)
        print("[步骤 4/4] 导出结果")
        print("=" * 80)

        # 1. 导出交叉验证详细结果
        all_cv_details = []
        for cv_result in self.cv_results:
            df_fold = self.cv.export_fold_details(cv_result)
            all_cv_details.append(df_fold)

        df_cv_all = pd.concat(all_cv_details, ignore_index=True)
        cv_path = self.output_dir / "cv_results_all_folds.csv"
        df_cv_all.to_csv(cv_path, index=False)
        print(f"  ✓ 交叉验证详细结果: {cv_path}")

        # 2. 导出交叉验证聚合结果
        cv_summary = []
        for cv_result in self.cv_results:
            row = {"model": cv_result.model_name}
            row.update(cv_result.aggregate_metrics)
            cv_summary.append(row)

        df_cv_summary = pd.DataFrame(cv_summary)
        cv_summary_path = self.output_dir / "cv_results_summary.csv"
        df_cv_summary.to_csv(cv_summary_path, index=False)
        print(f"  ✓ 交叉验证聚合结果: {cv_summary_path}")

        # 3. 导出样本外结果
        outsample_path = self.output_dir / "outsample_results.csv"
        self.outsample_df.to_csv(outsample_path, index=False)
        print(f"  ✓ 样本外评估结果: {outsample_path}")

        # 4. 生成markdown报告
        self._generate_markdown_report()

    def _generate_markdown_report(self):
        """生成markdown格式的训练报告"""
        report_path = self.output_dir / "training_report.md"

        with open(report_path, "w") as f:
            f.write("# 基线模型训练报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 数据信息
            f.write("## 数据信息\n\n")
            f.write(f"- 样本内数据: {self.X_insample.shape[0]} 样本 × {self.X_insample.shape[1]} 特征\n")
            f.write(f"- 样本外数据: {self.X_outsample.shape[0]} 样本 × {self.X_outsample.shape[1]} 特征\n")
            f.write(f"- 交叉验证: {self.n_folds}-fold\n\n")

            # 交叉验证结果
            f.write("## 交叉验证结果（样本内）\n\n")
            cv_summary = []
            for cv_result in self.cv_results:
                cv_summary.append({
                    "模型": cv_result.model_name,
                    "IC (均值)": f"{cv_result.aggregate_metrics['val_ic_pearson_mean']:.6f}",
                    "IC (标准差)": f"{cv_result.aggregate_metrics['val_ic_pearson_std']:.6f}",
                    "RMSE": f"{cv_result.aggregate_metrics['val_rmse_mean']:.6f}"
                })
            df_cv = pd.DataFrame(cv_summary)
            f.write(df_cv.to_markdown(index=False))
            f.write("\n\n")

            # 样本外结果
            f.write("## 样本外评估结果\n\n")
            df_out = self.outsample_df.sort_values("ic_pearson", ascending=False)
            df_display = df_out[["model", "ic_pearson", "ic_spearman", "rmse", "r2"]].copy()
            df_display.columns = ["模型", "IC (Pearson)", "IC (Spearman)", "RMSE", "R²"]
            f.write(df_display.to_markdown(index=False))
            f.write("\n\n")

            # 结论
            f.write("## 结论\n\n")
            best_model = df_out.iloc[0]["model"]
            best_ic = df_out.iloc[0]["ic_pearson"]
            f.write(f"- **最佳模型（样本外IC）**: {best_model} (IC = {best_ic:.6f})\n")
            f.write("- 所有模型均使用全量特征（无特征选择）\n")
            f.write("- 所有模型均使用固定超参数（未调优）\n\n")

            f.write("## 下一步\n\n")
            f.write("1. 在 `s03_hyperparameter_tuning/` 中运行超参数搜索\n")
            f.write("2. 在 `s04_feature_selection/` 中运行特征选择实验\n")

        print(f"  ✓ Markdown报告: {report_path}")


def main():
    """主函数"""
    trainer = BaselineModelTrainer(
        data_path="data/data.csv",
        output_dir="results/baseline_models",
        random_state=42,
        n_folds=4
    )

    trainer.run()


if __name__ == "__main__":
    main()
