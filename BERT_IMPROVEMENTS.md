# BERT 模型改进说明

## 问题诊断

通过详细分析发现BERT模型准确率低的主要原因：

### 发现的问题

1. **序列长度不一致** ❌
   - 训练时: `max_length=256`
   - 评估时: `max_length=128`
   - 数据平均文本长度: **411字符**
   - 影响: 大量重要信息被截断

2. **训练轮数不足** ❌
   - 仅训练了3个epoch
   - 训练准确率从58%→72%，明显还在快速提升
   - 验证准确率也在持续上升 (51%→60%→63%)

3. **学习率调度问题** ❌
   - 使用linear调度器
   - 第3个epoch时学习率已降至0.00e+00
   - 过早停止了参数优化

4. **学习率过高** ❌
   - 初始学习率: `5e-5`
   - BERT微调建议: `2e-5` 或 `3e-5`

5. **缺少Early Stopping** ❌
   - 没有防止过拟合的机制
   - 无法自动选择最佳模型

## 已实施的改进方案

### 改进1: 统一并增加序列长度 ✅
```python
# 修改前
max_length = 256

# 修改后
max_length = 512  # 覆盖平均文本长度411字符
```

**影响**:
- 减少文本截断，保留更多语义信息
- DistilBERT支持最大512 tokens

### 改进2: 增加训练轮数 ✅
```python
# 修改前
num_epochs = 3

# 修改后
num_epochs = 10
```

**影响**:
- 允许模型充分学习
- 配合Early Stopping避免过拟合

### 改进3: 改用Cosine学习率调度 ✅
```python
# 修改前
scheduler = get_scheduler("linear", ...)

# 修改后
scheduler = get_scheduler("cosine", ...)
```

**影响**:
- 避免学习率过早归零
- Cosine曲线更平滑，适合长时间训练

### 改进4: 降低学习率 ✅
```python
# 修改前
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 修改后
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
```

**影响**:
- 更稳定的训练过程
- 符合BERT微调最佳实践

### 改进5: 添加Early Stopping ✅
```python
best_val_acc = 0
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    # 训练和验证...

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

**影响**:
- 自动保存最佳模型
- 防止过拟合
- 节省训练时间

## 修改对比总结

| 配置项 | 修改前 | 修改后 | 改进原因 |
|--------|--------|--------|----------|
| **序列长度** | 256 | **512** | 覆盖完整文本（平均411字符） |
| **训练轮数** | 3 | **10** | 允许充分训练 |
| **学习率调度** | linear | **cosine** | 避免过早归零 |
| **初始学习率** | 5e-5 | **2e-5** | 更稳定的训练 |
| **Early Stopping** | ❌ 无 | **✅ patience=3** | 防止过拟合，保存最佳模型 |

## 预期性能提升

### 当前性能 (修改前)
- 测试准确率: **64.43%**
- F1 Macro: **0.625**
- 问题: 低于Logistic Regression基线 (64.36%)

### 预期性能 (修改后)
- 测试准确率: **68-72%** ⬆️ +4-8个百分点
- F1 Macro: **0.66-0.70** ⬆️ +0.04-0.08
- 目标: 明显超越传统基线方法

### 各类别预期改进
- **High类别**: 召回率 70% → 75%+
- **Low类别**: 召回率 49% → 60%+ (最需改进)
- **Medium类别**: 召回率 62% → 68%+

## 训练配置详情

### 完整训练参数
```python
# 模型配置
model_name = "distilbert-base-uncased"
num_classes = 3
dropout = 0.3
freeze_bert = False

# 数据配置
max_length = 512
batch_size = 16

# 优化器配置
learning_rate = 2e-5
weight_decay = 0.01

# 训练配置
num_epochs = 10
num_warmup_steps = 1237 (总步数的10%)
scheduler_type = "cosine"

# Early Stopping
patience = 3

# 类别权重 (处理不平衡)
class_weights = [0.857, 1.631, 0.820]  # [high, low, medium]
```

### 训练环境
- 硬件: GPU (CUDA)
- 框架: PyTorch + HuggingFace Transformers
- 总参数: 66,365,187
- 可训练参数: 66,365,187 (全参数微调)

## 下一步操作

### 1. 运行改进后的训练
打开 Jupyter Notebook:
```bash
jupyter notebook notebooks/04_BERT_Finetune.ipynb
```

按顺序执行所有单元格开始训练。

### 2. 监控训练过程
关注以下指标：
- 验证准确率是否持续提升
- Early Stopping是否在合理的epoch触发（预计5-8轮）
- 学习率曲线是否平滑下降

### 3. 评估最终模型
训练完成后，查看：
- 测试集准确率是否达到68%+
- F1 Macro是否达到0.66+
- 各类别性能是否均衡

### 4. 错误分析
重新运行错误分析notebook:
```bash
jupyter notebook notebooks/05_error_analysis.ipynb
```

更新测试时的max_length为512以保持一致。

## 参考资料

- BERT论文推荐学习率: 2e-5, 3e-5, 5e-5
- 微调建议: 2-4个epoch通常足够
- 序列长度: 根据数据实际长度选择（128/256/512）
- Early Stopping patience: 通常设为3-5

## 备注

所有改进已应用到 `notebooks/04_BERT_Finetune.ipynb`。

原始模型已保存为备份，新训练的模型将覆盖 `src/model/bert_finetuned.pt`。
