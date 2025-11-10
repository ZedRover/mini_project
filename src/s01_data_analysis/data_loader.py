#!/usr/bin/env python3
"""
数据加载和预处理模块
负责数据读取、清洗、划分样本内外数据
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """数据加载和预处理类"""

    def __init__(
        self,
        data_path: str | Path = "data/data.csv",
        target_column: str = "realY",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        初始化数据加载器

        Parameters
        ----------
        data_path : str | Path
            数据文件路径
        target_column : str
            目标变量列名
        test_size : float
            样本外数据比例（用于最终测试）
        random_state : int
            随机种子，保证可重现性
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        # 存储加载的数据
        self.df: pd.DataFrame | None = None
        self.X_insample: pd.DataFrame | None = None
        self.X_outsample: pd.DataFrame | None = None
        self.y_insample: pd.Series | None = None
        self.y_outsample: pd.Series | None = None

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        加载数据并划分样本内外数据

        Returns
        -------
        X_insample, X_outsample, y_insample, y_outsample
            样本内特征、样本外特征、样本内目标、样本外目标
        """
        print("=" * 70)
        print("数据加载与划分")
        print("=" * 70)

        # 1. 加载数据
        print(f"\n[1/3] 加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path, index_col=0)
        print(f"  ✓ 数据形状: {self.df.shape}")
        print(f"  ✓ 样本数: {self.df.shape[0]:,}")
        print(f"  ✓ 特征数: {self.df.shape[1] - 1:,}")

        # 2. 数据质量检查
        print(f"\n[2/3] 数据质量检查")
        missing_count = self.df.isnull().sum().sum()
        print(f"  ✓ 缺失值: {missing_count}")

        if missing_count > 0:
            print("  ⚠ 警告: 存在缺失值，将进行处理...")
            self.df = self.df.dropna()
            print(f"  ✓ 删除缺失值后形状: {self.df.shape}")

        # 3. 划分特征和目标变量
        if self.target_column not in self.df.columns:
            raise ValueError(f"目标变量 '{self.target_column}' 不存在于数据集中")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        print(f"  ✓ 目标变量统计:")
        print(f"    - 均值: {y.mean():.6f}")
        print(f"    - 标准差: {y.std():.6f}")
        print(f"    - 最小值: {y.min():.6f}")
        print(f"    - 最大值: {y.max():.6f}")

        # 4. 划分样本内外数据
        print(f"\n[3/3] 划分样本内外数据 (样本外比例: {self.test_size:.1%})")
        self.X_insample, self.X_outsample, self.y_insample, self.y_outsample = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )

        print(f"  ✓ 样本内数据: {self.X_insample.shape[0]:,} 样本")
        print(f"  ✓ 样本外数据: {self.X_outsample.shape[0]:,} 样本")
        print(f"  ✓ 样本内目标均值: {self.y_insample.mean():.6f}")
        print(f"  ✓ 样本外目标均值: {self.y_outsample.mean():.6f}")

        print("\n" + "=" * 70)
        print("数据加载完成！")
        print("=" * 70)

        return self.X_insample, self.X_outsample, self.y_insample, self.y_outsample

    def get_feature_names(self) -> list[str]:
        """获取特征名称列表"""
        if self.X_insample is not None:
            return list(self.X_insample.columns)
        elif self.df is not None:
            return [col for col in self.df.columns if col != self.target_column]
        else:
            raise ValueError("请先调用 load_and_split() 加载数据")

    def get_summary(self) -> dict:
        """获取数据摘要信息"""
        if self.df is None:
            raise ValueError("请先调用 load_and_split() 加载数据")

        return {
            "total_samples": len(self.df),
            "n_features": len(self.get_feature_names()),
            "insample_size": len(self.X_insample) if self.X_insample is not None else 0,
            "outsample_size": len(self.X_outsample) if self.X_outsample is not None else 0,
            "target_mean": self.df[self.target_column].mean(),
            "target_std": self.df[self.target_column].std(),
            "target_min": self.df[self.target_column].min(),
            "target_max": self.df[self.target_column].max(),
        }


if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader(
        data_path="../data/data.csv",
        target_column="realY",
        test_size=0.2,
        random_state=42
    )

    X_in, X_out, y_in, y_out = loader.load_and_split()

    print("\n数据摘要:")
    summary = loader.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
