# 完整工作流程指南

本项目已完成模块化重构，提供清晰的数据科学项目工作流程。

## 📁 项目结构

```
src/
├── s01_data_analysis/          # 数据分析
├── s02_model_training/         # 基线模型训练（全量特征，固定参数）
├── s03_hyperparameter_tuning/  # 超参数调优（LASSO和LightGBM）
├── s04_feature_selection/      # 特征选择与对比
└── s05_results_comparison/     # 结果综合对比
```

## 🔄 完整工作流程

### 步骤1: 基线模型训练（全量特征）

使用所有332个特征和固定超参数训练5个基线模型。

```bash
python -m src.s02_model_training.train_models
```

**训练的模型**：
- LinearRegression
- Ridge (alpha=1.0)
- Lasso (alpha=0.01)
- LightGBM (n_estimators=500, learning_rate=0.05, num_leaves=31)
- NeuralNetwork (hidden_layers=(100,50))

**输出**：
- `results/baseline_models/cv_results_all_folds.csv` - 所有fold的详细结果
- `results/baseline_models/cv_results_summary.csv` - 交叉验证聚合结果
- `results/baseline_models/outsample_results.csv` - 样本外评估结果
- `results/baseline_models/training_report.md` - Markdown格式报告

---

### 步骤2: 超参数调优

#### 2.1 LASSO Alpha调优

搜索最佳的L1正则化强度alpha（13个候选值）。

```bash
python -m src.s03_hyperparameter_tuning.lasso_analysis
```

**输出**：
- `results/lasso_analysis/lasso_ic_pearson_fold_matrix.csv` - IC × Fold矩阵
- `results/lasso_analysis/lasso_stability_metrics.csv` - 稳定性指标
- `results/lasso_analysis/best_alpha.txt` - 最佳alpha值

#### 2.2 LightGBM超参数调优

搜索最佳的树模型参数组合。

```bash
python -m src.s03_hyperparameter_tuning.lightgbm_tuning
```

**默认参数网格**：
- n_estimators: [100, 300, 500, 700]
- learning_rate: [0.01, 0.05, 0.1]
- num_leaves: [15, 31, 63]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]

**输出**：
- `results/lightgbm_tuning/lightgbm_grid_search_results.csv` - 所有组合的性能
- `results/lightgbm_tuning/best_params.txt` - 最佳参数组合
- `results/lightgbm_tuning/top10_param_combinations.csv` - Top-10组合

---

### 步骤3: 特征选择实验

#### 3.1 LASSO特征选择（使用最佳alpha）

```python
from src.s04_feature_selection.lasso_feature_selector import LassoFeatureSelector
from src.s01_data_analysis.data_loader import DataLoader

# 加载数据
loader = DataLoader("data/data.csv", "realY")
X_insample, _, y_insample, _ = loader.load_and_split()

# LASSO特征选择（自动使用最佳alpha）
selector = LassoFeatureSelector(alpha="auto")  # 自动从调优结果读取
features, importance = selector.train_and_select_features(
    X_insample, y_insample,
    top_k=100
)
selector.evaluate_selected_features(X_insample, y_insample)
selector.export_results()
```

**输出**：
- `results/feature_selection/lasso/lasso_feature_importance.csv`
- `results/feature_selection/lasso/lasso_selected_features.json`
- `results/feature_selection/lasso/lasso_performance_comparison.csv`

#### 3.2 LightGBM多比例特征选择

测试60%, 75%, 90%三种特征比例的性能。

```python
from src.s04_feature_selection.lightgbm_feature_selector import LightGBMFeatureSelector

# LightGBM多比例特征选择
selector = LightGBMFeatureSelector(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31
)

# 对比60%, 75%, 90%三种比例
ratio_results = selector.train_and_compare_ratios(
    X_insample, y_insample,
    ratios=[0.6, 0.75, 0.9]
)

selector.export_ratio_comparison_results()
```

**输出**：
- `results/feature_selection/lightgbm/lightgbm_ratio_comparison.csv`
- `results/feature_selection/lightgbm/lightgbm_features_60pct.json`
- `results/feature_selection/lightgbm/lightgbm_features_75pct.json`
- `results/feature_selection/lightgbm/lightgbm_features_90pct.json`

---

## 📊 关键功能详解

### 自动选择最佳Alpha

LASSO特征选择器支持自动读取超参数调优的结果：

```python
# 方式1：自动读取（推荐）
selector = LassoFeatureSelector(alpha="auto")

# 方式2：手动指定
selector = LassoFeatureSelector(alpha=0.001)
```

### 多比例特征对比

LightGBM选择器可以对比不同特征比例的性能：

```python
# 对比3种比例
ratios = [0.6, 0.75, 0.9]  # 60%, 75%, 90%特征

ratio_results = selector.train_and_compare_ratios(X, y, ratios=ratios)

# 结果包含：
# - baseline: 全量特征
# - ratio_0.6: 60%特征 (约199个)
# - ratio_0.75: 75%特征 (约249个)
# - ratio_0.9: 90%特征 (约299个)
```

---

## 📈 预期结果

### 基线模型性能（参考）

| 模型 | IC (Pearson) | RMSE |
|------|-------------|------|
| LightGBM | 0.556 ± 0.019 | 0.438 |
| Lasso | 0.254 ± 0.021 | 0.506 |
| Ridge | 0.292 ± 0.020 | 0.497 |
| LinearRegression | 0.292 ± 0.020 | 0.497 |
| NeuralNetwork | ~0.25 | ~0.51 |

### LASSO特征选择效果（参考）

- 全量特征（332个）：IC = 0.254
- 筛选后（230个）：IC = 0.268 (+5.7%)
- 最佳alpha：0.001

### LightGBM特征选择效果（参考）

- 全量特征（332个）：IC = 0.557
- 筛选后（100个）：IC = 0.645 (+15.8%)
- 60%特征（199个）：IC ≈ 0.63
- 75%特征（249个）：IC ≈ 0.64
- 90%特征（299个）：IC ≈ 0.65

---

## 🔧 高级用法

### 自定义LASSO超参数网格

```python
from src.s03_hyperparameter_tuning.lasso_analysis import LassoAnalyzer

analyzer = LassoAnalyzer(
    alphas=[0.0001, 0.001, 0.01, 0.1, 1.0],  # 自定义alpha列表
    n_folds=4
)
analyzer.run_grid_search(X, y)
best_alpha, best_score = analyzer.get_best_alpha()
analyzer.export_results()
```

### 自定义LightGBM参数网格

```python
from src.s03_hyperparameter_tuning.lightgbm_tuning import LightGBMTuner

tuner = LightGBMTuner(
    param_grid={
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.03, 0.05, 0.1],
        'num_leaves': [31, 63],
        'subsample': [0.8, 1.0]
    }
)
tuner.run_grid_search(X, y)
best_params, best_score = tuner.get_best_params()
```

---

## 📝 注意事项

1. **虚拟环境**：确保使用项目虚拟环境
   ```bash
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

2. **运行顺序**：
   - 步骤1（基线训练）可以独立运行
   - 步骤2（超参数调优）可以独立运行，但建议先运行
   - 步骤3（特征选择）依赖步骤2的结果（如果使用`alpha="auto"`）

3. **内存占用**：
   - 超参数网格搜索可能需要较长时间（根据网格大小）
   - LightGBM调优时可设置`max_combinations`限制组合数

4. **结果目录**：
   - 所有结果默认保存在`results/`目录
   - Git已配置忽略results/*（除README.md）

---

## 🚀 快速开始

```bash
# 1. 训练基线模型
python -m src.s02_model_training.train_models

# 2. LASSO超参数调优
python -m src.s03_hyperparameter_tuning.lasso_analysis

# 3. LightGBM超参数调优（可选，耗时较长）
python -m src.s03_hyperparameter_tuning.lightgbm_tuning

# 4. 查看results/目录下的所有结果
ls -R results/
```

---

## 💡 常见问题

**Q: 如何修改交叉验证折数？**
A: 所有模块都支持`n_folds`参数，默认为4。

**Q: 特征选择一定要先做超参数调优吗？**
A: LASSO使用`alpha="auto"`时需要，否则可以手动指定alpha值。

**Q: 为什么LightGBM的性能远好于LASSO？**
A: LightGBM是基于树的非线性模型，能捕捉特征间的复杂交互，而LASSO是线性模型。

**Q: 60%/75%/90%比例是如何确定的？**
A: 这些是常用的特征选择比例，你可以自定义任何比例列表。

---

更多详情请参考：
- [MIGRATION_NOTES.md](MIGRATION_NOTES.md) - 迁移指南
- [src/README.md](src/README.md) - 模块详细说明
