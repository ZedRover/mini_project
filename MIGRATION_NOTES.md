# 代码重组说明

## 本次重组内容

### 目录结构变更

**之前**：所有Python文件在项目根目录
```
mini_project/
├── data_loader.py
├── train_models.py
├── lasso_trainer.py
└── ...（16个Python文件）
```

**现在**：按功能模块化组织在src/目录下
```
mini_project/
└── src/
    ├── s01_data_analysis/        # 数据分析模块
    ├── s02_model_training/        # 模型训练模块（全量特征）
    ├── s03_hyperparameter_tuning/ # 超参数调优
    ├── s04_feature_selection/     # 特征选择
    └── s05_results_comparison/    # 结果对比
```

### 文件重命名

- `lasso_trainer.py` → `lasso_feature_selector.py`（更准确地反映其功能）
- `lightgbm_trainer.py` → `lightgbm_feature_selector.py`

### 已完成的工作

- ✅ 创建模块化目录结构（s01-s05前缀保持逻辑顺序）
- ✅ 移动所有文件到对应目录
- ✅ 更新所有import语句适配新结构
- ✅ 创建__init__.py使每个目录成为Python包
- ✅ 删除过时的modeling_pipeline.py

## 后续需要完成的工作

### 1. 简化train_models.py（高优先级）

**当前问题**：train_models.py包含了LASSO alpha grid search逻辑

**需要修改**：
- 移除LASSO alpha grid search部分
- 只保留使用固定alpha的基线模型训练
- LASSO grid search应该完全在s03_hyperparameter_tuning/lasso_analysis.py中完成

### 2. 增强lasso_feature_selector.py

**需要添加**：
- 从lasso_analysis.py的结果中读取最佳alpha
- 或者作为参数传入最佳alpha
- 使用最佳alpha进行特征选择

### 3. 增强lightgbm_feature_selector.py

**需要添加**：
- 支持按比例选择特征：60%, 75%, 90%
- 对比不同比例的性能
- 生成比例对比报告和可视化

### 4. 重构run_feature_selection.py

**新的工作流程**：
```python
1. 运行全量特征训练（调用train_models.py或直接集成）
2. 运行LASSO特征选择（使用最佳alpha）
3. 运行LightGBM多比例特征选择（60%, 75%, 90%）
4. 对比分析：
   - 全量 vs 筛选特征
   - LASSO vs LightGBM
   - LightGBM不同比例对比
5. 生成综合报告
```

## 使用指南

### 当前可用的命令

```bash
# 激活虚拟环境
source .venv/bin/activate

# 或使用uv运行
uv run python -m src.s02_model_training.train_models
```

### 导入示例

```python
# 从其他模块导入
from src.s01_data_analysis.data_loader import DataLoader
from src.s02_model_training.metrics import MetricsCalculator
from src.s03_hyperparameter_tuning.lasso_analysis import LassoAnalyzer
```

## 技术细节

### Python模块命名
- 使用`s01`, `s02`等前缀（s代表step/stage）
- 不能使用`01`, `02`（Python不允许数字开头的模块名）

### Import路径
- 所有import使用绝对路径：`from src.sXX_module.file import Class`
- 从项目根目录运行所有脚本

## 下一步计划

1. **立即**：提交当前重组结构
2. **接下来**：
   - 简化train_models.py
   - 增强特征选择模块
   - 实现多比例实验
   - 测试完整流程
3. **最后**：更新文档和使用示例

## 兼容性注意事项

- 旧的导入语句将不再工作
- 需要从项目根目录运行脚本
- 虚拟环境(.venv)仍然可用

## 问题与解决方案

### Q: 为什么使用s01而不是01?
A: Python模块名不能以数字开头，s代表step/stage

### Q: 如何运行脚本?
A: 使用Python模块方式：`python -m src.s02_model_training.train_models`

### Q: 导入失败怎么办?
A: 确保从项目根目录运行，且Python能找到src/目录
