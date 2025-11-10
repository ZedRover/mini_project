#!/usr/bin/env python3
"""
唯一值详细分析
针对realY只有20个唯一值的特殊情况进行深入分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 加载数据
df = pd.read_csv('data/data.csv', index_col=0)
y = df['realY']

print("\n" + "="*80)
print("realY 唯一值详细分析")
print("="*80 + "\n")

# 获取所有唯一值
unique_values = sorted(y.unique())
print(f"唯一值数量: {len(unique_values)}\n")

# 创建唯一值分析表
value_analysis = []
for val in unique_values:
    count = (y == val).sum()
    percentage = count / len(y) * 100
    value_analysis.append({
        '序号': len(value_analysis) + 1,
        '数值': val,
        '出现次数': count,
        '占比(%)': percentage
    })

df_analysis = pd.DataFrame(value_analysis)
print("唯一值分布详情:")
print("="*80)
print(df_analysis.to_string(index=False))
print("="*80 + "\n")

# 分析唯一值之间的间隔
diffs = np.diff(unique_values)
print("唯一值之间的间隔分析:")
print(f"  最小间隔: {diffs.min():.6f}")
print(f"  最大间隔: {diffs.max():.6f}")
print(f"  平均间隔: {diffs.mean():.6f}")
print(f"  间隔标准差: {diffs.std():.6f}")
print(f"  间隔的变异系数: {diffs.std()/diffs.mean():.4f}\n")

# 判断是否等间隔
is_uniform = diffs.std() / diffs.mean() < 0.1
if is_uniform:
    print("✓ 唯一值近似等间隔分布\n")
else:
    print("✗ 唯一值间隔不均匀\n")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 唯一值分布条形图
ax = axes[0, 0]
bars = ax.bar(range(len(df_analysis)), df_analysis['出现次数'],
              color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('唯一值序号')
ax.set_ylabel('出现次数')
ax.set_title('20个唯一值的频次分布', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 2. 唯一值数值分布
ax = axes[0, 1]
ax.scatter(range(len(unique_values)), unique_values, s=100,
          c='red', marker='o', edgecolors='black', linewidths=2)
ax.plot(range(len(unique_values)), unique_values, 'b--', alpha=0.5)
ax.set_xlabel('序号')
ax.set_ylabel('数值大小')
ax.set_title('唯一值的数值分布', fontweight='bold')
ax.grid(alpha=0.3)

# 3. 间隔分布
ax = axes[1, 0]
ax.bar(range(len(diffs)), diffs, color='coral', alpha=0.7, edgecolor='black')
ax.axhline(diffs.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {diffs.mean():.4f}')
ax.set_xlabel('区间序号')
ax.set_ylabel('间隔大小')
ax.set_title('相邻唯一值的间隔', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. 占比饼图（前10个）
ax = axes[1, 1]
top_10 = df_analysis.nlargest(10, '占比(%)')
others_pct = df_analysis.iloc[10:]['占比(%)'].sum() if len(df_analysis) > 10 else 0
labels = [f'值{i+1}' for i in range(len(top_10))]
if others_pct > 0:
    labels.append('其他')
    values = list(top_10['占比(%)']) + [others_pct]
else:
    values = list(top_10['占比(%)'])

colors = plt.cm.Set3(range(len(values)))
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('占比分布（前10个唯一值）', fontweight='bold')

plt.tight_layout()
plt.savefig('results/target_analysis/unique_values_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 保存可视化: unique_values_analysis.png\n")
plt.close()

# 论证为什么仍然选择回归
print("="*80)
print("为什么仍然选择回归任务？")
print("="*80 + "\n")

print("尽管realY只有20个唯一值，但回归任务仍然是最佳选择，理由如下：\n")

print("1. 【数值有序性】")
print(f"   - 20个唯一值具有明确的大小关系：从{unique_values[0]:.3f}到{unique_values[-1]:.3f}")
print(f"   - 这些值不是随意标签，而是有实际数值意义的量")
print(f"   - 回归模型可以利用这种顺序信息，分类模型则会忽略\n")

print("2. 【距离信息有意义】")
print(f"   - 相邻值之间的距离是有意义的（平均间隔{diffs.mean():.4f}）")
print(f"   - 预测值2.0比预测值1.0更接近真实值1.5")
print(f"   - 分类模型无法表达这种\"接近程度\"\n")

print("3. 【类别间过渡平滑】")
print(f"   - 如果视为分类问题，则有20个类别")
print(f"   - 类别之间的边界是模糊的，强行分类会产生硬边界")
print(f"   - 回归的连续预测更符合数据的本质\n")

print("4. 【评估指标更准确】")
print(f"   - IC（信息系数）可以评估预测的排序能力")
print(f"   - RMSE可以量化预测误差的大小")
print(f"   - 分类准确率无法区分\"差一点\"和\"差很多\"\n")

print("5. 【模型泛化性更好】")
print(f"   - 回归模型可以预测介于两个唯一值之间的结果")
print(f"   - 这在新数据上可能出现（样本外数据的真实值可能不在这20个值中）")
print(f"   - 分类模型只能预测已见过的20个类别\n")

print("="*80)
print("结论")
print("="*80)
print("""
虽然训练集的realY只有20个唯一值，但这并不改变其数值本质。
这20个值之间存在明确的大小关系和距离信息，这正是回归任务的核心。

将其视为分类问题会：
- 忽略数值之间的顺序关系
- 丢失距离信息
- 在类别边界处产生不合理的硬切分
- 无法准确评估预测误差的程度

因此，回归任务仍然是明智的、科学的、符合数据本质的选择。
""")
print("="*80)

# 保存分析结果
df_analysis.to_csv('results/target_analysis/unique_values_detail.csv', index=False)
print("\n✓ 保存详细数据: unique_values_detail.csv")
