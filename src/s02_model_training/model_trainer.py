#!/usr/bin/env python3
"""
模型训练器
定义并训练多种回归模型
"""

from __future__ import annotations

from typing import Dict

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ModelFactory:
    """模型工厂类"""

    def __init__(self, random_state: int = 42):
        """
        初始化模型工厂

        Parameters
        ----------
        random_state : int
            随机种子
        """
        self.random_state = random_state

    def get_linear_regression(self) -> Pipeline:
        """获取线性回归模型（带标准化）"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    def get_ridge(self, alpha: float = 1.0) -> Pipeline:
        """获取Ridge回归模型（带标准化）"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=self.random_state))
        ])

    def get_lasso(self, alpha: float = 0.001) -> Pipeline:
        """
        获取LASSO回归模型（带标准化）

        Parameters
        ----------
        alpha : float
            L1正则化强度
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=alpha,
                max_iter=10000,
                random_state=self.random_state,
                tol=1e-4
            ))
        ])

    def get_lightgbm(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ) -> LGBMRegressor:
        """
        获取LightGBM回归模型

        Parameters
        ----------
        n_estimators : int
            树的数量
        learning_rate : float
            学习率
        num_leaves : int
            叶子节点数量
        subsample : float
            样本采样比例
        colsample_bytree : float
            特征采样比例
        """
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True
        )

    def get_neural_network(
        self,
        hidden_layers: tuple = (128, 64, 32),
        learning_rate: float = 0.001,
        max_iter: int = 300
    ) -> Pipeline:
        """
        获取神经网络回归模型（带标准化）

        Parameters
        ----------
        hidden_layers : tuple
            隐藏层结构
        learning_rate : float
            学习率
        max_iter : int
            最大迭代次数
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                learning_rate_init=learning_rate,
                max_iter=max_iter,
                batch_size=256,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state,
                verbose=False
            ))
        ])

    def get_all_baseline_models(self) -> Dict[str, BaseEstimator]:
        """
        获取所有基线模型

        Returns
        -------
        Dict[str, BaseEstimator]
            模型名称到模型对象的映射
        """
        return {
            "LinearRegression": self.get_linear_regression(),
            "Ridge": self.get_ridge(alpha=1.0),
            "Lasso": self.get_lasso(alpha=0.001),
            "LightGBM": self.get_lightgbm(),
            "NeuralNetwork": self.get_neural_network()
        }

    def get_lasso_grid(self, alphas: list[float] = None) -> Dict[str, BaseEstimator]:
        """
        获取多个alpha值的LASSO模型

        Parameters
        ----------
        alphas : list[float], optional
            alpha值列表，默认为 [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

        Returns
        -------
        Dict[str, BaseEstimator]
            模型名称到模型对象的映射
        """
        if alphas is None:
            # 使用对数尺度的alpha值
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

        models = {}
        for alpha in alphas:
            # 使用科学计数法格式化模型名称
            if alpha < 0.01:
                name = f"Lasso_alpha_{alpha:.0e}"
            else:
                name = f"Lasso_alpha_{alpha:.2f}"
            models[name] = self.get_lasso(alpha=alpha)

        return models


def print_model_info(model: BaseEstimator) -> None:
    """
    打印模型信息

    Parameters
    ----------
    model : BaseEstimator
        模型对象
    """
    print(f"模型类型: {type(model).__name__}")

    if isinstance(model, Pipeline):
        print("Pipeline步骤:")
        for name, step in model.steps:
            print(f"  - {name}: {type(step).__name__}")
        # 打印最后一步的参数
        final_step = model.steps[-1][1]
        print(f"\n{type(final_step).__name__} 参数:")
        for key, value in final_step.get_params().items():
            print(f"  {key}: {value}")
    else:
        print("模型参数:")
        for key, value in model.get_params().items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # 测试模型工厂
    factory = ModelFactory(random_state=42)

    print("=" * 70)
    print("基线模型")
    print("=" * 70)

    models = factory.get_all_baseline_models()
    for name, model in models.items():
        print(f"\n{name}:")
        print_model_info(model)

    print("\n" + "=" * 70)
    print("LASSO网格搜索模型")
    print("=" * 70)

    lasso_grid = factory.get_lasso_grid()
    for name in lasso_grid.keys():
        print(f"  - {name}")
