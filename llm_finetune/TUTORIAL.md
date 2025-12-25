# LoRA 与 LlamaFactory 教程

## 第一部分：LoRA 基础

### 什么是 LoRA？

**LoRA (Low-Rank Adaptation)** 是一种高效的大模型微调方法。

#### 传统微调 vs LoRA

| 方法 | 训练参数量 | 显存需求 | 训练时间 | 模型大小 |
|------|-----------|----------|----------|---------|
| **全量微调** | 7B | ~60GB | 很长 | 14GB |
| **LoRA** | 3-20M | ~24GB | 快 | 300-800MB |

**原理：**

```
原始模型权重 W (固定不变)
     ↓
W + ΔW = W + B×A
     ↑
LoRA 权重（只训练 B 和 A）

其中：B 和 A 是小矩阵
rank = A 的列数 = B 的行数
```

---

### LoRA 核心参数详解

#### 1. `lora_rank` (秩)

**定义：** LoRA 分解矩阵的维度

```python
# 原始权重矩阵 W: [d, d]  (例如 4096×4096)
# LoRA 分解：
# W_new = W + B×A
# B: [d, rank]
# A: [rank, d]

# rank=8:  B=[4096, 8], A=[8, 4096]  → 参数量小
# rank=64: B=[4096, 64], A=[64, 4096] → 参数量大
```

**选择指南：**

| Rank | 参数量 | 适用场景 | 效果 |
|------|--------|----------|------|
| 8 | ~3M | 快速实验、资源受限 | ⭐⭐ |
| 16 | ~6M | 简单任务 | ⭐⭐⭐ |
| 32 | ~12M | 一般任务 | ⭐⭐⭐⭐ |
| **64** | ~24M | **复杂任务（推荐）** | ⭐⭐⭐⭐⭐ |
| 128 | ~48M | 接近全量微调 | ⭐⭐⭐⭐⭐ |

**建议：**
- 数据量 <5K: rank=16-32
- 数据量 5K-20K: rank=32-64
- 数据量 >20K: rank=64-128

---

#### 2. `lora_alpha` (缩放因子)

**定义：** 控制 LoRA 权重的影响力

```python
# LoRA 更新公式：
W_new = W + (alpha / rank) × B×A

# alpha=32, rank=16:
# 缩放系数 = 32/16 = 2

# alpha=128, rank=64:
# 缩放系数 = 128/64 = 2
```

**经验法则：**

```yaml
# 标准配置（推荐）
lora_alpha: 2 × lora_rank

# 示例：
lora_rank: 16  →  lora_alpha: 32
lora_rank: 32  →  lora_alpha: 64
lora_rank: 64  →  lora_alpha: 128
lora_rank: 128 →  lora_alpha: 256
```

**调整建议：**
- 保持 `alpha/rank = 2` 通常效果最好
- 增大 alpha：LoRA 影响更大（可能过拟合）
- 减小 alpha：LoRA 影响更小（可能欠拟合）

---

#### 3. `lora_dropout`

**定义：** 防止过拟合的正则化方法

```yaml
lora_dropout: 0.1  # 随机丢弃 10% 的 LoRA 权重
```

**选择指南：**

| Dropout | 适用场景 |
|---------|----------|
| 0.0 | 数据量很大（>50K） |
| 0.05 | 数据量大（20K-50K） |
| **0.1** | **一般情况（推荐）** |
| 0.2 | 数据量小（<5K）或严重过拟合 |

---

#### 4. `lora_target` (目标模块)

**定义：** 对哪些层应用 LoRA

```yaml
# 选项1：只训练注意力层（快速）
lora_target: q_proj,v_proj

# 选项2：训练所有注意力（常用）
lora_target: q_proj,k_proj,v_proj,o_proj

# 选项3：训练所有线性层（推荐，效果最好）
lora_target: all
```

**Transformer 层结构：**

```
Transformer Block
├── Self-Attention
│   ├── q_proj (Query)   ← LoRA
│   ├── k_proj (Key)     ← LoRA
│   ├── v_proj (Value)   ← LoRA
│   └── o_proj (Output)  ← LoRA
└── Feed-Forward
    ├── gate_proj        ← LoRA
    ├── up_proj          ← LoRA
    └── down_proj        ← LoRA
```

**建议：**
- **快速实验**: `q_proj,v_proj` (最少)
- **平衡**: `q_proj,k_proj,v_proj,o_proj`
- **最佳效果**: `all` (推荐，A800 充裕)

---

### LoRA 参数配置示例

#### 轻量级配置

```yaml
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target: q_proj,v_proj

# 适用：快速实验、显存受限
# 参数量：~6M
# 显存：~12GB
```

#### 平衡配置

```yaml
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.1
lora_target: q_proj,k_proj,v_proj,o_proj

# 适用：一般任务
# 参数量：~12M
# 显存：~18GB
```

#### 推荐配置（A800）

```yaml
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all

# 适用：复杂任务、充足显存
# 参数量：~24M
# 显存：~24GB
```

#### 大 Rank 配置

```yaml
lora_rank: 128
lora_alpha: 256
lora_dropout: 0.1
lora_target: all

# 适用：追求极致效果
# 参数量：~48M
# 显存：~35GB
```

---

## 第二部分：训练超参数

### 1. 学习率 (`learning_rate`)

**定义：** 优化器的步长

```yaml
learning_rate: 2.0e-4  # 0.0002
```

**LoRA 学习率指南：**

| 学习率 | 适用场景 | 风险 |
|--------|----------|------|
| 1e-4 | 保守、稳定 | 收敛慢 |
| **2e-4** | **LoRA 推荐** | ✓ |
| 3e-4 | 激进 | 可能不稳定 |
| 5e-4 | 很大 | 容易不收敛 |

**注意：** LoRA 的学习率通常比全量微调大 10 倍
- 全量微调：1e-5 ~ 5e-5
- LoRA：1e-4 ~ 5e-4

---

### 2. 训练轮数 (`num_train_epochs`)

```yaml
num_train_epochs: 3
```

**选择指南：**

| Epochs | 数据量 | 说明 |
|--------|--------|------|
| 1-2 | >50K | 数据量大，避免过拟合 |
| **3** | 10K-50K | **推荐** |
| 3-5 | 5K-10K | 数据量中等 |
| 5-10 | <5K | 数据量小 |

**判断依据：**
- 看 loss 曲线：如果 val_loss 不再下降，停止
- 看评估指标：如果验证集性能不再提升，停止

---

### 3. Batch Size 与梯度累积

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 8

# 有效 batch size = 4 × 8 = 32
```

**显存 vs Batch Size：**

| GPU 显存 | Batch Size | 梯度累积 | 有效 Batch |
|----------|-----------|----------|-----------|
| 24GB | 2 | 16 | 32 |
| 40GB | 4 | 8 | 32 |
| **80GB (A800)** | **8** | **4** | **32** |

**建议：**
- 有效 batch size 保持在 32-64 之间
- 显存不足时，减小 batch size，增大梯度累积

---

### 4. 学习率调度器 (`lr_scheduler_type`)

```yaml
lr_scheduler_type: cosine
```

**可选类型：**

| 类型 | 学习率变化 | 适用场景 |
|------|-----------|----------|
| `linear` | 线性衰减 | 简单任务 |
| **`cosine`** | **余弦衰减（推荐）** | **大多数任务** |
| `constant` | 不变 | 很少用 |

**Cosine 调度示例：**

```
LR
 ↑
 │     ┌─────┐
 │    /       \
 │   /         \
 │  /           \___
 │ /
 └──────────────────→ Steps
   预热   稳定    衰减
```

---

### 5. 预热 (`warmup_ratio`)

```yaml
warmup_ratio: 0.1  # 前 10% 步数预热
```

**作用：** 训练初期逐渐增大学习率，避免梯度爆炸

```
LR
 ↑
 │        ┌─────────
 │       /
 │      /
 │     /   预热阶段
 │    /
 └────────────────→ Steps
    10%
```

**建议：**
- 小数据集：0.03 ~ 0.05
- 大数据集：0.1 ~ 0.2
- 默认：0.1

---

### 6. 权重衰减 (`weight_decay`)

```yaml
weight_decay: 0.01
```

**作用：** L2 正则化，防止过拟合

| 值 | 适用场景 |
|----|----------|
| 0.0 | 数据量很大 |
| **0.01** | **推荐** |
| 0.1 | 严重过拟合 |

---

## 第三部分：LlamaFactory 使用

### 基本用法

#### 1. 命令行训练

```bash
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml
```

#### 2. Web UI

```bash
llamafactory-cli webui
# 访问 http://localhost:7860
```

#### 3. 命令行覆盖参数

```bash
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --lora_rank 32 \
    --learning_rate 3e-4 \
    --output_dir ./outputs/experiment1
```

---

### 配置文件结构

```yaml
### 模型
model_name_or_path: Qwen/Qwen2-7B-Instruct

### 方法
stage: sft                    # 监督微调
finetuning_type: lora         # LoRA 方法

### LoRA 参数
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all

### 数据集
dataset: ticket_multi_task_train
dataset_dir: ./data
template: qwen                # Qwen 模型专用模板
cutoff_len: 2048             # 最大序列长度

### 训练参数
output_dir: ./outputs/qwen2-7b-ticket-lora
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1

### 优化
bf16: true                   # BF16 混合精度
flash_attn: fa2              # Flash Attention 2
gradient_checkpointing: true # 梯度检查点（省显存）

### 评估
val_size: 0.05              # 5% 数据作为验证集
eval_strategy: steps
eval_steps: 200             # 每 200 步评估一次

### 日志
logging_steps: 10
save_steps: 500
save_total_limit: 3         # 只保留最近 3 个检查点
```

---

### 常用命令

```bash
# 训练
llamafactory-cli train config.yaml

# 导出合并后的模型
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --adapter_name_or_path ./outputs/qwen2-7b-ticket-lora \
    --export_dir ./models/qwen2-7b-ticket-merged \
    --template qwen \
    --finetuning_type lora

# 聊天测试
llamafactory-cli chat \
    --model_name_or_path ./outputs/qwen2-7b-ticket-lora \
    --template qwen

# Web UI
llamafactory-cli webui
```

---

## 第四部分：参数调优实践

### 实验设计

#### 实验1：Rank 对比

| 实验 | Rank | Alpha | 预期效果 |
|------|------|-------|---------|
| A | 16 | 32 | 基线 |
| B | 32 | 64 | 更好 |
| C | 64 | 128 | 推荐 |
| D | 128 | 256 | 最佳 |

```bash
llamafactory-cli train configs/qwen2_7b_lora_rank16.yaml
llamafactory-cli train configs/qwen2_7b_lora_rank32.yaml
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml
llamafactory-cli train configs/qwen2_7b_lora_rank128.yaml
```

---

#### 实验2：学习率对比

```bash
# LR = 1e-4
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --learning_rate 1e-4 --output_dir ./outputs/lr_1e4

# LR = 2e-4
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --learning_rate 2e-4 --output_dir ./outputs/lr_2e4

# LR = 5e-4
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --learning_rate 5e-4 --output_dir ./outputs/lr_5e4
```

---

#### 实验3：Target 对比

```bash
# 只训练 QV
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --lora_target q_proj,v_proj \
    --output_dir ./outputs/target_qv

# 训练 QKVO
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --lora_target q_proj,k_proj,v_proj,o_proj \
    --output_dir ./outputs/target_qkvo

# 训练所有层
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --lora_target all \
    --output_dir ./outputs/target_all
```

---

### 监控与调试

#### TensorBoard

```bash
tensorboard --logdir ./outputs --port 6006
```

**查看指标：**
- `train/loss`: 训练损失（应该持续下降）
- `eval/loss`: 验证损失（如果上升 → 过拟合）
- `train/learning_rate`: 学习率变化
- `train/grad_norm`: 梯度范数（太大 → 爆炸）

---

#### 常见问题诊断

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| Loss 不降 | 学习率太小 | 增大 LR |
| Loss 震荡 | 学习率太大 | 减小 LR |
| Val loss 上升 | 过拟合 | 增加 dropout / 减少 epochs |
| OOM 错误 | 显存不足 | 减小 batch size |
| Grad norm 很大 | 梯度爆炸 | 减小 LR / 检查数据 |

---

## 第五部分：参数速查表

### LoRA 参数

| 参数 | 推荐值 | 范围 | 说明 |
|------|--------|------|------|
| `lora_rank` | 64 | 8-128 | 越大效果越好 |
| `lora_alpha` | 128 | rank×2 | 缩放因子 |
| `lora_dropout` | 0.1 | 0-0.2 | 防过拟合 |
| `lora_target` | all | - | 训练的层 |

### 训练参数

| 参数 | 推荐值 | 范围 | 说明 |
|------|--------|------|------|
| `learning_rate` | 2e-4 | 1e-4 ~ 5e-4 | LoRA 专用 |
| `num_train_epochs` | 3 | 1-10 | 训练轮数 |
| `per_device_train_batch_size` | 4 | 1-16 | 单卡 batch |
| `gradient_accumulation_steps` | 8 | 1-32 | 梯度累积 |
| `warmup_ratio` | 0.1 | 0.03-0.2 | 预热比例 |
| `lr_scheduler_type` | cosine | - | 调度器 |
| `weight_decay` | 0.01 | 0-0.1 | 权重衰减 |

### 优化选项

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `bf16` | true | BF16 混合精度 |
| `flash_attn` | fa2 | Flash Attention 2 |
| `gradient_checkpointing` | true | 省显存 |

---

## 总结

### 推荐配置（A800 80GB）

```yaml
# LoRA
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all

# 训练
learning_rate: 2.0e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
warmup_ratio: 0.1
lr_scheduler_type: cosine
weight_decay: 0.01

# 优化
bf16: true
flash_attn: fa2
gradient_checkpointing: true
```

### 调参顺序

1. **先调 LoRA rank**：找到效果最好的 rank
2. **再调学习率**：在最佳 rank 下调 LR
3. **最后调其他**：batch size、epochs 等

### 实验建议

- **最少 3 组对比**：rank16, rank64, rank128
- **记录所有指标**：loss, ROUGE, accuracy, 训练时间
- **使用 TensorBoard**：可视化对比
- **自动生成报告**：使用评估脚本（见下一节）
