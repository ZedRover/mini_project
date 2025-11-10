#!/usr/bin/env python3
"""Comprehensive EDA pipeline tailored for wide tabular datasets."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"


class DetailedEDA:
    """End-to-end exploratory data analysis helper."""

    def __init__(
        self,
        data_path: str | Path = "data.csv",
        index_col: int = 0,
        target_column: Optional[str] = None,
        output_dir: str | Path = "eda_plots",
    ) -> None:
        self.data_path = Path(data_path)
        self.index_col = index_col
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = 42

        self.df: pd.DataFrame | None = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []

    # ------------------------------------------------------------------ #
    # Data ingestion and metadata
    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame | None:
        """Load the dataset from disk."""
        print("=" * 60)
        print("1. Data Loading")
        print("=" * 60)

        try:
            self.df = pd.read_csv(self.data_path, index_col=self.index_col)
            print(f"✓ Loaded data from {self.data_path}")
            print(f"✓ Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
            memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
            print(f"✓ Memory usage: {memory_mb:.2f} MB")
            return self.df
        except Exception as exc:  # noqa: BLE001 - surface full context
            print(f"✗ Failed to load data: {exc}")
            self.df = None
            return None

    def basic_info(self) -> None:
        """Print dataset headline information."""
        if self.df is None:
            print("Load data before requesting basic info.")
            return

        print("\n" + "=" * 60)
        print("2. Basic Dataset Info")
        print("=" * 60)

        print(f"Dataset shape: {self.df.shape}")
        print(f"Row count: {self.df.shape[0]:,}")
        print(f"Column count: {self.df.shape[1]:,}")

        print("\nColumn names skipped (sequential X1..Xn pattern assumed).")

        print("\nDtype summary:")
        for dtype, count in self.df.dtypes.value_counts().items():
            print(f"  {dtype}: {count} columns")

        self._identify_column_types()

        print(f"\nNumeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        print(f"Datetime columns: {len(self.datetime_columns)}")

    def _identify_column_types(self) -> None:
        """Populate column type lists."""
        if self.df is None:
            self.numeric_columns = []
            self.categorical_columns = []
            self.datetime_columns = []
            return

        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.datetime_columns = self.df.select_dtypes(include=["datetime64"]).columns.tolist()

    # ------------------------------------------------------------------ #
    # Quality, statistics, and tabular summaries
    # ------------------------------------------------------------------ #
    def data_quality_check(self) -> Dict[str, Any]:
        """Assess missing values, duplicates, and uniqueness."""
        if self.df is None:
            print("Load data before running quality checks.")
            return {}

        print("\n" + "=" * 60)
        print("3. Data Quality")
        print("=" * 60)

        quality_report: Dict[str, Any] = {}

        missing_df = self.df.isnull().sum().to_frame(name="missing_count")
        missing_df["missing_pct"] = missing_df["missing_count"] / len(self.df) * 100
        missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)

        print("Missing value summary:")
        if missing_df.empty:
            print("  ✓ No missing values detected.")
        else:
            print(missing_df.head(20))
        quality_report["missing_values"] = missing_df.to_dict()

        duplicate_rows = self.df.duplicated().sum()
        duplicate_pct = duplicate_rows / len(self.df) * 100
        print(f"\nDuplicate rows: {duplicate_rows} ({duplicate_pct:.2f}% of the dataset)")
        quality_report["duplicate_rows"] = duplicate_rows

        unique_df = self.df.nunique().to_frame(name="unique_count")
        unique_df["unique_pct"] = unique_df["unique_count"] / len(self.df) * 100
        print("\nUnique value summary (top 20 columns by uniqueness):")
        print(unique_df.sort_values("unique_pct", ascending=False).head(20))
        quality_report["unique_values"] = unique_df.to_dict()

        return quality_report

    def descriptive_statistics(self) -> Dict[str, Any]:
        """Display descriptive stats for numeric and categorical fields."""
        if self.df is None:
            print("Load data before running descriptive statistics.")
            return {}

        print("\n" + "=" * 60)
        print("4. Descriptive Statistics")
        print("=" * 60)

        stats_report: Dict[str, Any] = {}

        if self.numeric_columns:
            numeric_stats = self.df[self.numeric_columns].describe().transpose()
            extra_stats = pd.DataFrame(
                {
                    "skewness": self.df[self.numeric_columns].skew(),
                    "kurtosis": self.df[self.numeric_columns].kurtosis(),
                    "coefficient_of_variation": self.df[self.numeric_columns].std()
                    / (self.df[self.numeric_columns].mean().replace(0, np.nan)),
                }
            )
            print("Numeric summary (first 10 columns shown):")
            print(numeric_stats.head(10))
            stats_report["numeric"] = {
                "describe": numeric_stats.to_dict(),
                "extra_stats": extra_stats.to_dict(),
            }
        else:
            print("No numeric columns detected.")

        if self.categorical_columns:
            print("\nCategorical summary:")
            categorical_summary: Dict[str, Any] = {}
            for col in self.categorical_columns:
                value_counts = self.df[col].value_counts(dropna=False)
                categorical_summary[col] = {
                    "unique_count": int(value_counts.shape[0]),
                    "most_frequent": value_counts.index[0],
                    "most_frequent_count": int(value_counts.iloc[0]),
                    "top_values": value_counts.head(5).to_dict(),
                }
                print(f"  {col}: {categorical_summary[col]}")
            stats_report["categorical"] = categorical_summary
        else:
            print("\nNo categorical columns detected.")

        return stats_report

    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    def create_visualizations(self, save_plots: bool = True) -> None:
        """Generate static and interactive plots."""
        if self.df is None:
            print("Load data before creating visualizations.")
            return

        print("\n" + "=" * 60)
        print("5. Visualizations")
        print("=" * 60)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.numeric_columns:
            self._visualize_numeric_columns(save_plots)
            self._create_correlation_heatmap(save_plots)
            self._plot_dimensionality_reduction(save_plots)
        else:
            print("No numeric columns available for distribution or correlation plots.")

        if self.categorical_columns:
            self._visualize_categorical_columns(save_plots)

        if self.numeric_columns and self.target_column:
            self._plot_target_relationships(save_plots)

    def _numeric_sample(self, max_rows: int = 4000) -> pd.DataFrame:
        assert self.df is not None  # for type checkers
        numeric_df = self.df[self.numeric_columns]
        if len(numeric_df) > max_rows:
            return numeric_df.sample(max_rows, random_state=self.random_state)
        return numeric_df

    def _save_figure(self, fig: plt.Figure, filename: Optional[str], save_plots: bool) -> None:
        if save_plots and filename:
            path = self.output_dir / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"    saved figure -> {path}")
        plt.close(fig)

    def _visualize_numeric_columns(self, save_plots: bool) -> None:
        """Plot representative numeric distributions and boxplots."""
        assert self.df is not None
        sample_df = self._numeric_sample()
        variance_rank = sample_df.var().sort_values(ascending=False)
        hist_columns = variance_rank.head(min(12, len(variance_rank))).index.tolist()

        if not hist_columns:
            print("No numeric columns available for histogram batches.")
            return

        print(f"Selected {len(hist_columns)} high-variance columns for histograms "
              f"out of {len(self.numeric_columns)} numeric columns.")

        batch_size = 3
        for batch_idx in range(0, len(hist_columns), batch_size):
            batch_columns = hist_columns[batch_idx : batch_idx + batch_size]
            fig, axes = plt.subplots(1, len(batch_columns), figsize=(5 * len(batch_columns), 4))
            if len(batch_columns) == 1:
                axes = [axes]  # type: ignore[assignment]
            for ax, col in zip(axes, batch_columns):
                ax.hist(sample_df[col], bins=40, color="steelblue", alpha=0.8, edgecolor="white")
                ax.set_title(col, fontsize=11)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.grid(alpha=0.25)
            plt.tight_layout()
            self._save_figure(fig, f"numeric_hist_batch_{batch_idx // batch_size + 1}.png", save_plots)

        # Boxplots leverage the same high-variance subset to reduce clutter.
        boxplot_batch_size = 6
        for batch_idx in range(0, len(hist_columns), boxplot_batch_size):
            batch_columns = hist_columns[batch_idx : batch_idx + boxplot_batch_size]
            fig, ax = plt.subplots(figsize=(10, max(4, len(batch_columns) * 0.5)))
            sns.boxplot(data=sample_df[batch_columns], orient="h", ax=ax, palette="viridis")
            ax.set_title(f"Numeric boxplots batch {batch_idx // boxplot_batch_size + 1}")
            ax.set_xlabel("Value")
            plt.tight_layout()
            self._save_figure(fig, f"numeric_box_batch_{batch_idx // boxplot_batch_size + 1}.png", save_plots)

    def _visualize_categorical_columns(self, save_plots: bool) -> None:
        """Plot categorical distributions with top categories only."""
        assert self.df is not None
        print("Categorical distributions:")
        for idx, col in enumerate(self.categorical_columns, start=1):
            value_counts = self.df[col].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(value_counts.index.astype(str), value_counts.values, color="salmon")
            ax.set_title(f"{col} distribution (top {len(value_counts)})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            for bar, val in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            self._save_figure(fig, f"categorical_{idx}_{col}.png", save_plots)

    def _create_correlation_heatmap(self, save_plots: bool) -> None:
        """Render a correlation heatmap with automatic column curation."""
        assert self.df is not None
        correlation_matrix = self.df[self.numeric_columns].corr()
        corr_columns = list(correlation_matrix.columns)

        if len(corr_columns) < 2:
            print("Not enough numeric columns for a correlation heatmap.")
            return

        if len(corr_columns) > 30:
            avg_abs_corr = correlation_matrix.abs().mean().sort_values(ascending=False)
            corr_columns = avg_abs_corr.head(30).index.tolist()
            correlation_matrix = correlation_matrix.loc[corr_columns, corr_columns]
            print(f"Using the top {len(corr_columns)} columns with the strongest average correlations for the heatmap.")

        fig, ax = plt.subplots(figsize=(min(1.2 * len(corr_columns), 18), min(1.2 * len(corr_columns), 18)))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap="coolwarm",
            center=0,
            annot=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            ax=ax,
        )
        ax.set_title("Correlation heatmap (filtered)")
        plt.tight_layout()
        self._save_figure(fig, "correlation_heatmap.png", save_plots)

        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if pd.notna(corr_value) and abs(float(corr_value)) > 0.7:
                    high_corr_pairs.append(
                        (correlation_matrix.columns[i], correlation_matrix.columns[j], float(corr_value))
                    )

        if high_corr_pairs:
            print("\nHighly correlated pairs (|r| > 0.7):")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
                print(f"  {col1} vs {col2}: {corr:.3f}")
        else:
            print("\nNo feature pairs exceeded |r| > 0.7.")

    def _plot_target_relationships(self, save_plots: bool) -> None:
        """Summarize how numeric features relate to the target column."""
        assert self.df is not None
        if not self.target_column or self.target_column not in self.df.columns:
            print("Target column not available; skipping target relationship plots.")
            return

        numeric_features = [col for col in self.numeric_columns if col != self.target_column]
        if not numeric_features:
            print("No numeric features aside from the target.")
            return

        corr_series = self.df[numeric_features].corrwith(self.df[self.target_column]).dropna()
        if corr_series.empty:
            print("Unable to compute correlations with the target column.")
            return

        ranked = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
        top_corr = ranked.head(15)
        print(f"\nTop correlations with {self.target_column}:")
        print(top_corr.to_string())

        fig, ax = plt.subplots(figsize=(8, max(4, len(top_corr) * 0.4)))
        sns.barplot(x=top_corr.values, y=top_corr.index, palette="viridis", ax=ax)
        ax.set_title(f"Features most correlated with {self.target_column}")
        ax.set_xlabel("Correlation coefficient")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        self._save_figure(fig, f"target_corr_{self.target_column}.png", save_plots)

        if save_plots:
            sample_size = min(len(self.df), 2000)
            sampled_df = self.df.sample(sample_size, random_state=self.random_state)
            parallel_dims = top_corr.index[:6].tolist()
            if parallel_dims:
                fig_parallel = px.parallel_coordinates(
                    sampled_df[parallel_dims + [self.target_column]],
                    color=self.target_column,
                    color_continuous_scale="Viridis",
                    labels={dim: dim for dim in parallel_dims + [self.target_column]},
                    title=f"Parallel coordinates (top {len(parallel_dims)} features + {self.target_column})",
                )
                path = self.output_dir / f"parallel_coordinates_{self.target_column}.html"
                fig_parallel.write_html(path, include_plotlyjs="cdn")
                print(f"    saved interactive figure -> {path}")

    def _plot_dimensionality_reduction(self, save_plots: bool) -> None:
        """Use PCA to project the feature space for large tables."""
        assert self.df is not None
        if len(self.numeric_columns) < 3:
            print("Not enough numeric columns for dimensionality reduction.")
            return

        numeric_df = self.df[self.numeric_columns].dropna()
        if numeric_df.empty:
            print("No complete rows available for PCA projection.")
            return

        if len(numeric_df) > 5000:
            numeric_df = numeric_df.sample(5000, random_state=self.random_state)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2, random_state=self.random_state)
        components = pca.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        if self.target_column and self.target_column in self.df.columns:
            target_aligned = self.df.loc[numeric_df.index, self.target_column]
            scatter = ax.scatter(
                components[:, 0],
                components[:, 1],
                c=target_aligned,
                cmap="viridis",
                s=18,
                alpha=0.7,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(self.target_column)
        else:
            ax.scatter(components[:, 0], components[:, 1], color="steelblue", s=18, alpha=0.6)

        ax.set_title("PCA projection (first two components)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save_figure(fig, "pca_projection.png", save_plots)

    # ------------------------------------------------------------------ #
    # Diagnostics and reporting
    # ------------------------------------------------------------------ #
    def outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using IQR and Z-score rules."""
        if self.df is None:
            print("Load data before running outlier detection.")
            return {}

        print("\n" + "=" * 60)
        print("6. Outlier Detection")
        print("=" * 60)

        if not self.numeric_columns:
            print("No numeric columns available for outlier detection.")
            return {}

        outlier_report: Dict[str, Any] = {}
        summary_rows: List[Dict[str, float]] = []

        for col in self.numeric_columns:
            series = self.df[col].dropna()
            if series.empty:
                continue

            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]

            z_scores = np.abs((series - series.mean()) / series.std(ddof=0))
            outliers_zscore = series[z_scores > 3]

            iqr_pct = len(outliers_iqr) / len(series) * 100
            z_pct = len(outliers_zscore) / len(series) * 100

            outlier_report[col] = {
                "iqr": {
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_count": int(len(outliers_iqr)),
                    "outlier_pct": iqr_pct,
                    "sample_values": outliers_iqr.head(10).tolist(),
                },
                "zscore": {
                    "outlier_count": int(len(outliers_zscore)),
                    "outlier_pct": z_pct,
                    "sample_values": outliers_zscore.head(10).tolist(),
                },
            }

            summary_rows.append({"column": col, "iqr_pct": iqr_pct, "zscore_pct": z_pct})

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values("iqr_pct", ascending=False).head(10)
            print("Columns with the highest IQR-based outlier ratios:")
            print(summary_df.round(3).to_string(index=False))
        else:
            print("No outliers detected using the configured rules.")

        return outlier_report

    def generate_recommendations(self) -> List[str]:
        """Generate preprocessing recommendations."""
        if self.df is None:
            print("Load data before generating recommendations.")
            return []

        print("\n" + "=" * 60)
        print("7. Data Preparation Recommendations")
        print("=" * 60)

        recommendations: List[str] = []

        missing_values = self.df.isnull().sum()
        high_missing_cols = missing_values[missing_values > len(self.df) * 0.3].index.tolist()
        if high_missing_cols:
            recommendations.append(
                f"Columns with >30% missing values may need to be dropped or imputed: {high_missing_cols}"
            )

        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicated rows to avoid leakage.")

        for col in self.categorical_columns:
            unique_count = self.df[col].nunique()
            if unique_count > len(self.df) * 0.5:
                recommendations.append(
                    f"{col} has {unique_count} unique values (>50% of rows). Consider grouping or hashing."
                )

        if self.numeric_columns:
            skewness = self.df[self.numeric_columns].skew()
            skewed_cols = skewness[skewness.abs() > 2].index.tolist()
            if skewed_cols:
                recommendations.append(
                    f"Columns with |skew| > 2 could benefit from log/Box-Cox transformations: {skewed_cols}"
                )

            correlation_matrix = self.df[self.numeric_columns].corr().abs()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if pd.notna(corr_val) and corr_val > 0.9:
                        high_corr_pairs.append(
                            (correlation_matrix.columns[i], correlation_matrix.columns[j], float(corr_val))
                        )
            if high_corr_pairs:
                recommendations.append(
                    f"Strongly collinear pairs detected (|r| > 0.9); consider feature selection: {high_corr_pairs[:5]}"
                )

        if recommendations:
            for idx, rec in enumerate(recommendations, start=1):
                print(f"{idx}. {rec}")
        else:
            print("No critical data quality issues detected.")

        return recommendations

    def save_report(self, filename: str | None = None) -> None:
        """Persist a concise text report."""
        if self.df is None:
            print("Load data before saving a report.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eda_report_{timestamp}.txt"

        print(f"\nSaving report to {filename}")

        numeric_preview = self.df[self.numeric_columns].describe().head(5) if self.numeric_columns else None
        missing_summary = self.df.isnull().sum()

        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("Detailed Exploratory Data Analysis Report\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write(f"Source file: {self.data_path}\n")
            handle.write(f"Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns\n\n")

            handle.write("Column overview\n")
            handle.write("-" * 30 + "\n")
            handle.write(f"Numeric columns: {len(self.numeric_columns)}\n")
            handle.write(f"Categorical columns: {len(self.categorical_columns)}\n")
            handle.write(f"Datetime columns: {len(self.datetime_columns)}\n\n")

            handle.write("Missing values\n")
            handle.write("-" * 30 + "\n")
            handle.write(f"{missing_summary[missing_summary > 0]}\n\n")

            if numeric_preview is not None:
                handle.write("Numeric preview\n")
                handle.write("-" * 30 + "\n")
                handle.write(f"{numeric_preview}\n\n")

        print(f"✓ Report stored at {filename}")

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_full_analysis(self, save_plots: bool = True, save_report: bool = True) -> None:
        """Execute the entire EDA workflow in order."""
        print("Starting detailed EDA...")
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.load_data() is None:
            return

        self.basic_info()
        self.data_quality_check()
        self.descriptive_statistics()
        self.create_visualizations(save_plots)
        self.outlier_detection()
        self.generate_recommendations()

        if save_report:
            self.save_report()

        print("\n" + "=" * 60)
        print("EDA run completed.")
        print("=" * 60)


def main() -> None:
    """Entrypoint for running the module directly."""
    eda = DetailedEDA(data_path="data.csv", index_col=0, target_column="realY")
    eda.run_full_analysis(save_plots=True, save_report=True)


if __name__ == "__main__":
    main()
