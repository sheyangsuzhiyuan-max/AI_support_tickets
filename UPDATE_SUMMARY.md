# BERT实验脚本重大更新

## 更新内容

### 1. 自动生成完整可视化报告

`run_personal_project.py` 现在会自动生成：

#### 输出文件：
- **`BERT_Finetuning_Report.md`** - 根目录的完整实验报告（可放简历）
- **`figures/training_curves.png`** - 训练曲线对比图
- **`figures/confusion_matrix.png`** - 最佳模型的混淆矩阵
- **`figures/performance_comparison.png`** - 性能对比柱状图
- **`data/bert_experiments_*.json`** - 详细实验结果（JSON）
- **`data/bert_experiments_summary_*.csv`** - 结果摘要（CSV）

#### 报告内容包括：
1. Executive Summary - 最佳配置摘要
2. Experiment Overview - 实验概览
3. Results Summary - 结果对比表格
4. Training Curves - 训练曲线可视化
5. Performance Comparison - 性能对比图表
6. Confusion Matrix - 混淆矩阵热力图
7. Detailed Analysis - 详细分析和排名
8. Conclusions and Recommendations - 结论和生产建议

### 2. 新增功能

- ✅ 自动计算最佳模型的confusion matrix
- ✅ 生成classification report（precision, recall, F1 per class）
- ✅ 训练过程可视化（loss, accuracy, F1 curves）
- ✅ 实验间性能对比可视化
- ✅ Markdown格式报告，适合GitHub展示和简历
- ✅ 高分辨率图表（300 DPI，适合打印）

### 3. 使用方法

```bash
# 运行所有实验（5个实验：不同LR、epoch、freeze策略）
python run_personal_project.py

# 快速测试模式（1 epoch）
python run_personal_project.py --quick

# 运行特定实验
python run_personal_project.py --exp EXP1 EXP5
```

### 4. 输出示例

运行完成后会在根目录生成：
- `BERT_Finetuning_Report.md` - 包含所有分析和图表引用
- `figures/` 目录 - 包含所有可视化图表
- `data/` 目录 - 包含原始数据

报告文件可以直接：
- 放到GitHub项目README展示
- 添加到简历的项目链接
- 用于面试时的技术展示

## 技术改进

### 新增依赖
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
```

### 新增函数
1. `generate_visualizations(all_results, output_dir='figures')` - 生成所有图表
2. `generate_markdown_report(all_results, summary_df, figure_paths, timestamp)` - 生成Markdown报告
3. `run_experiment()` 新增参数 `compute_confusion=False` - 可选计算混淆矩阵

### 图表样式
- 使用 seaborn 主题
- 高分辨率输出 (300 DPI)
- 专业配色方案
- 清晰的标题和标签

## 与Assignment的区别

| 特性 | Assignment (`run_assignment.py`) | Personal Project (`run_personal_project.py`) |
|------|----------------------------------|---------------------------------------------|
| 目标 | 生成作业报告 | 系统实验对比 |
| 模型数量 | 3个（LogReg, CNN, BERT） | 5个BERT实验 |
| 可视化 | 基础表格 | 完整图表+混淆矩阵 |
| 报告格式 | 学术风格 | 技术展示风格 |
| 输出位置 | `CA6000_Assignment_Report.md` | `BERT_Finetuning_Report.md` |
| 适用场景 | 课程提交 | 简历/GitHub展示 |

## 下一步

1. 在服务器运行 `python run_personal_project.py`
2. 等待实验完成（约2-3小时，取决于GPU）
3. 检查生成的 `BERT_Finetuning_Report.md`
4. 查看 `figures/` 目录的可视化图表
5. 将报告和图表添加到Git仓库
6. 在简历中展示这个项目！
