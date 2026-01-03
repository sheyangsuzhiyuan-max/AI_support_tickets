# Comprehensive Model Analysis Report
# 智能工单系统 - 模型评估综合分析报告

**Project:** AI Support Ticket Classification & Response Generation
**Base Model:** Qwen2-7B-Instruct
**Fine-tuning Method:** LoRA (Low-Rank Adaptation)
**Test Dataset:** 4,240 samples
**Report Generated:** 2025-12-31

---

## Executive Summary

本报告对比分析了 Base Model 与三种不同 LoRA Rank 配置的 Fine-tuned 模型，论证了微调的必要性并确定了最优超参数配置。

### Key Results

| Model | Type Acc | Priority Acc | Queue Acc | ROUGE-L | BLEU |
|-------|----------|--------------|-----------|---------|------|
| **Base Model** | 49.91% | 33.68% | 29.76% | N/A | N/A |
| LoRA Rank=32 | 81.27% | 46.01% | 40.68% | 0.3567 | 0.1495 |
| LoRA Rank=64 | 83.33% | 51.82% | 48.96% | 0.3966 | 0.1969 |
| **LoRA Rank=128** | **85.40%** | **57.10%** | **54.06%** | **0.4555** | **0.2610** |

### Key Findings

1. **Fine-tuning 必要性验证:** Base Model 在所有分类任务上接近随机水平，无法满足生产需求
2. **最优 LoRA Rank:** Rank=128 在所有指标上均表现最佳
3. **性能提升:** 相比 Base Model，最优配置实现 +71% Type 准确率提升

---

## Part 1: Why Fine-tuning? Base Model 问题分析

### 1.1 Base Model 性能评估

| Task | Base Model | Random Baseline | Gap |
|------|------------|-----------------|-----|
| Type Classification (4类) | 49.91% | 25.00% | +24.91pp |
| Priority Classification (3类) | 33.68% | 33.33% | +0.35pp |
| Queue Classification (8类) | 29.76% | 12.50% | +17.26pp |

**分析:**
- **Type (49.91%):** 接近50%说明模型只能区分部分类别，无法准确分类
- **Priority (33.68%):** 几乎等于随机猜测(33.3%)，模型无法判断优先级
- **Queue (29.76%):** 虽高于随机基线(12.5%)，但8分类任务54%+的错误率无法接受

### 1.2 Base Model 输出问题

#### 问题1: 输出格式不符合要求

| 期望格式 | Base Model 实际输出 |
|----------|---------------------|
| `Classification:` | `1. **Classification:**` |
| `- Type: Incident` | `   - Type: Problem` |
| `Response:` | `2. **Professional Response:**` |

#### 问题2: 输出过于冗长

| Model | Avg Output Length | Reference Length | Ratio |
|-------|-------------------|------------------|-------|
| Base Model | 319.59 words | 57.92 words | **5.52x** |
| Fine-tuned | 58.76 words | 57.92 words | 1.01x |

#### 问题3: 风格不适合客服场景

Base Model 输出特点:
- 大量 Markdown 格式（标题、粗体、列表）
- 详细的步骤说明和时间承诺
- 冗长的开场和结尾

期望的客服回复:
- 简洁直接
- 快速响应客户问题
- 标准化格式便于处理

### 1.3 Fine-tuning 必要性结论

| 问题 | Base Model | Fine-tuning 解决方案 |
|------|------------|---------------------|
| 分类准确率低 | Type 49.91%, Priority 33.68% | 通过领域数据学习分类模式 |
| 输出格式混乱 | 自由格式，无法解析 | 学习标准化输出格式 |
| 回复过长 | 320词/条 | 控制在60词左右 |
| 风格不匹配 | 通用助手风格 | 专业客服风格 |

**结论:** Base Model 无法直接用于工单处理，Fine-tuning 是必要的。

---

## Part 2: LoRA Rank 对比实验

### 2.1 实验设置

| 参数 | 值 |
|------|-----|
| Base Model | Qwen2-7B-Instruct |
| LoRA Target Modules | all (所有线性层) |
| LoRA Alpha | 2 × Rank |
| LoRA Dropout | 0.1 |
| Learning Rate | 2e-4 |
| LR Scheduler | cosine |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |
| Training Epochs | 3 |
| Batch Size | 4 |
| Gradient Accumulation | 8 |
| Effective Batch Size | 32 |
| Max Sequence Length | 2048 |
| Max Training Samples | 20,000 |
| Precision | bf16 |
| Test Samples | 4,240 |

### 2.2 分类任务对比

#### Type Classification (4类: Incident, Request, Problem, Change)

| Model | Accuracy | vs Base | vs Rank=32 |
|-------|----------|---------|------------|
| Base Model | 49.91% | - | - |
| Rank=32 | 81.27% | +62.83% | - |
| Rank=64 | 83.33% | +66.95% | +2.53% |
| **Rank=128** | **85.40%** | **+71.12%** | **+5.08%** |

#### Priority Classification (3类: high, medium, low)

| Model | Accuracy | vs Base | vs Rank=32 |
|-------|----------|---------|------------|
| Base Model | 33.68% | - | - |
| Rank=32 | 46.01% | +36.60% | - |
| Rank=64 | 51.82% | +53.87% | +12.63% |
| **Rank=128** | **57.10%** | **+69.54%** | **+24.10%** |

#### Queue Classification (8类)

| Model | Accuracy | vs Base | vs Rank=32 |
|-------|----------|---------|------------|
| Base Model | 29.76% | - | - |
| Rank=32 | 40.68% | +36.70% | - |
| Rank=64 | 48.96% | +64.55% | +20.35% |
| **Rank=128** | **54.06%** | **+81.62%** | **+32.89%** |

### 2.3 生成质量对比

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|-------|---------|---------|---------|------|
| Base Model | N/A | N/A | N/A | N/A |
| Rank=32 | 0.4826 | 0.2496 | 0.3567 | 0.1495 |
| Rank=64 | 0.5142 | 0.2935 | 0.3966 | 0.1969 |
| **Rank=128** | **0.5639** | **0.3573** | **0.4555** | **0.2610** |

**ROUGE-L 提升趋势:**
```
Rank=32  → Rank=64:  +11.19% (0.3567 → 0.3966)
Rank=64  → Rank=128: +14.85% (0.3966 → 0.4555)
Rank=32  → Rank=128: +27.70% (0.3567 → 0.4555)
```

### 2.4 综合性能得分

为了综合评估，我们计算加权得分:

**权重分配:**
- Type Accuracy: 25%
- Priority Accuracy: 25%
- Queue Accuracy: 25%
- ROUGE-L: 25%

| Model | Type | Priority | Queue | ROUGE-L | **Weighted Score** |
|-------|------|----------|-------|---------|-------------------|
| Base Model | 49.91 | 33.68 | 29.76 | 0* | 28.34 |
| Rank=32 | 81.27 | 46.01 | 40.68 | 35.67 | 50.91 |
| Rank=64 | 83.33 | 51.82 | 48.96 | 39.66 | 55.94 |
| **Rank=128** | **85.40** | **57.10** | **54.06** | **45.55** | **60.53** |

*Base Model ROUGE-L 因格式问题无法计算，按0计

---

## Part 3: 最优参数分析

### 3.1 Rank vs 性能曲线

```
Classification Accuracy Trend:

85% ─────────────────────────────────●─── Rank=128 (85.40%)
    │                           ╱
    │                       ╱
83% ─────────────────────●─────────────── Rank=64 (83.33%)
    │                ╱
    │            ╱
81% ───────────●───────────────────────── Rank=32 (81.27%)
    │
    │
50% ──●────────────────────────────────── Base (49.91%)
    │
    └──────┬──────┬──────┬──────┬──────
          Base   32     64    128   Rank


ROUGE-L Trend:

0.46 ─────────────────────────────────●─── Rank=128 (0.4555)
     │                           ╱
     │                       ╱
0.40 ─────────────────────●─────────────── Rank=64 (0.3966)
     │                ╱
     │            ╱
0.36 ───────────●───────────────────────── Rank=32 (0.3567)
     │
     └──────┬──────┬──────┬──────┬──────
           Base   32     64    128   Rank
```

### 3.2 Rank 选择分析

| Rank | 可训练参数 | Type Acc | ROUGE-L | 边际收益 |
|------|-----------|----------|---------|----------|
| 32 | ~13M | 81.27% | 0.3567 | Baseline |
| 64 | ~26M | 83.33% | 0.3966 | +2.06pp Type, +0.04 ROUGE |
| 128 | ~52M | 85.40% | 0.4555 | +2.07pp Type, +0.06 ROUGE |

**分析:**
- **Rank 32→64:** 参数量翻倍，Type +2.06pp, ROUGE-L +11.2%
- **Rank 64→128:** 参数量翻倍，Type +2.07pp, ROUGE-L +14.9%
- **结论:** Rank=128 边际收益仍然显著，未出现收益递减

### 3.3 最优配置推荐

| 使用场景 | 推荐 Rank | 理由 |
|----------|-----------|------|
| **生产环境** | **128** | 最佳性能，显存占用可接受 |
| 资源受限 | 64 | 平衡性能与资源 |
| 快速原型 | 32 | 训练最快，基础效果 |

**最终推荐:** LoRA Rank=128

---

## Part 4: 结论与建议

### 4.1 主要结论

1. **Fine-tuning 必要性 ✅**
   - Base Model 分类准确率接近随机水平 (33-50%)
   - 输出格式和长度无法控制
   - Fine-tuning 后所有指标显著提升

2. **最优 LoRA Rank: 128 ✅**
   - Type Accuracy: 85.40% (最高)
   - Priority Accuracy: 57.10% (最高)
   - Queue Accuracy: 54.06% (最高)
   - ROUGE-L: 0.4555 (最高)

3. **性能提升总结**
   | 指标 | Base → Rank=128 | 提升幅度 |
   |------|-----------------|----------|
   | Type Acc | 49.91% → 85.40% | +71.12% |
   | Priority Acc | 33.68% → 57.10% | +69.54% |
   | Queue Acc | 29.76% → 54.06% | +81.62% |
   | Output Length | 320词 → 59词 | -81.6% |

### 4.2 目标达成情况

| 指标 | 目标 | 达成值 | 状态 |
|------|------|--------|------|
| Type Accuracy | 80% | 85.40% | ✅ 超额完成 |
| ROUGE-L | 0.40 | 0.4555 | ✅ 超额完成 |
| Priority Accuracy | 85% | 57.10% | ⚠️ 未达成 |
| Queue Accuracy | 70% | 54.06% | ⚠️ 未达成 |

### 4.3 进一步优化建议

1. **Priority 分类优化 (当前 57.10%)**
   - 检查训练数据中 priority 标签一致性
   - 考虑增加 priority 判断的上下文特征
   - 可能需要更清晰的 priority 定义标准

2. **Queue 分类优化 (当前 54.06%)**
   - 8分类任务较难，考虑层级分类策略
   - 分析易混淆类别，针对性增加训练样本
   - 可考虑引入 queue 描述的额外特征

3. **模型架构探索**
   - 尝试 Rank=256 观察是否有进一步提升
   - 考虑 QLoRA 减少显存占用
   - 探索全量微调作为上限参考

---

## Appendix: 详细数据

### A. Base Model 评估结果
```json
{
  "type_accuracy": 0.4990566037735849,
  "queue_accuracy": 0.2976415094339623,
  "priority_accuracy": 0.33679245283018866,
  "avg_pred_length": 319.59,
  "avg_ref_length": 57.92
}
```

### B. LoRA Rank=32 评估结果
```json
{
  "avg_rouge1_f": 0.4826,
  "avg_rouge2_f": 0.2496,
  "avg_rougeL_f": 0.3567,
  "avg_bleu": 0.1495,
  "type_accuracy": 0.8127,
  "queue_accuracy": 0.4068,
  "priority_accuracy": 0.4601
}
```

### C. LoRA Rank=64 评估结果
```json
{
  "avg_rouge1_f": 0.5142,
  "avg_rouge2_f": 0.2935,
  "avg_rougeL_f": 0.3966,
  "avg_bleu": 0.1969,
  "type_accuracy": 0.8333,
  "queue_accuracy": 0.4896,
  "priority_accuracy": 0.5182
}
```

### D. LoRA Rank=128 评估结果
```json
{
  "avg_rouge1_f": 0.5639,
  "avg_rouge2_f": 0.3573,
  "avg_rougeL_f": 0.4555,
  "avg_bleu": 0.2610,
  "type_accuracy": 0.8540,
  "queue_accuracy": 0.5406,
  "priority_accuracy": 0.5710
}
```

---

**Report End**
