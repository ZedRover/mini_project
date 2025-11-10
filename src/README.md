# 项目代码结构说明

## 目录组织

项目按照数据科学工作流程的逻辑顺序组织代码：

### s01_data_analysis/ - 数据分析
- `data_loader.py` - 数据加载与样本内外划分
- `data_prep.py` - 数据预处理
- `detailed_eda.py` - 详细的探索性数据分析
- `target_variable_analysis.py` - 目标变量分析
- `unique_values_analysis.py` - 唯一值深度分析

### s02_model_training/ - 模型训练（全量特征）
- `cross_validation.py` - 4-fold交叉验证框架
- `metrics.py` - 评估指标（IC, RMSE, 分位数分析等）
- `model_trainer.py` - 模型工厂（创建各种模型）
- `train_models.py` - **主训练脚本：使用全量特征训练所有模型**

### s03_hyperparameter_tuning/ - 超参数调优
- `lasso_analysis.py` - LASSO alpha网格搜索
- `visualization.py` - 可视化工具（热图、稳定性分析等）

### s04_feature_selection/ - 特征选择
- `lasso_feature_selector.py` - 使用最佳alpha进行LASSO特征选择
- `lightgbm_feature_selector.py` - LightGBM特征选择（60%/75%/90%比例对比）
- `feature_selection_comparison.py` - 特征选择结果对比分析
- `run_feature_selection.py` - **主特征选择脚本：先全量训练，再特征筛选对比**

### s05_results_comparison/ - 结果对比
- 综合结果对比与分析

## 工作流程

1. **数据加载** (`s01_data_analysis/data_loader.py`)
   - 加载data.csv
   - 划分样本内80% / 样本外20%

2. **模型训练** (`s02_model_training/train_models.py`)
   - 使用全部332个特征训练基线模型：
     - LinearRegression
     - Ridge
     - LASSO (使用固定alpha)
     - LightGBM
   - 4-fold交叉验证
   - 记录IC、RMSE等指标

3. **超参数调优** (`s03_hyperparameter_tuning/lasso_analysis.py`)
   - LASSO alpha网格搜索（13个候选值）
   - 分析不同alpha对IC的影响
   - 选择最佳alpha

4. **特征选择** (`s04_feature_selection/run_feature_selection.py`)
   - **LASSO特征选择**：
     - 使用最佳alpha训练
     - 基于系数绝对值排序
     - 选择Top-K特征
   - **LightGBM特征选择**：
     - 基于gain特征重要性
     - 测试60%, 75%, 90%特征比例
     - 对比不同比例的性能
   - 特征重叠分析：Jaccard相似度、Venn图等
   - 跨模型性能测试

5. **结果对比**
   - 全量特征 vs 筛选特征
   - 不同模型性能对比
   - 可视化报告

## 使用方法

```bash
# 1. 训练基线模型（全量特征）
python -m src.s02_model_training.train_models

# 2. LASSO超参数调优
python -m src.s03_hyperparameter_tuning.lasso_analysis

# 3. 特征选择实验
python -m src.s04_feature_selection.run_feature_selection
```

## 关键改进

- ✅ 清晰的模块化分离
- ✅ 逻辑顺序编号（s01, s02, s03...）
- ✅ 全量特征训练与特征选择解耦
- ✅ 支持多比例特征选择实验
- ✅ 完整的性能对比框架
