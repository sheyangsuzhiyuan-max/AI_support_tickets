# 智能工单系统 - LLM 微调项目

## 项目概述

基于 **Qwen2-7B + LoRA** 微调，构建智能客服工单系统，实现：
- 🏷️ 自动分类（type/queue/priority）
- 💬 智能回复生成
- ✅ 人工审核界面
- 📊 工单数据看板

---

## 快速开始

### 本地准备数据

```bash
cd llm_finetune/scripts
python prepare_data.py --task_type multi_task
```

### 上传到服务器（Git）

```bash
git init
git add .
git commit -m "Initial commit"
git push
```

### 服务器配置

```bash
ssh username@server
cd /mnt/kai_ckp/alex
git clone https://your-repo.git
cd 000_ai_support_tickets/llm_finetune

# 一键配置环境（自动检测路径）
bash scripts/setup_server.sh
bash scripts/prepare_training.sh
```

### 开始训练

```bash
# 批量运行3个rank对比实验（推荐）
bash scripts/run_rank_comparison.sh

# 或单个实验
cd /mnt/kai_ckp/alex/LLaMA-Factory
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml
```

### 评估与报告

```bash
# 批量评估并自动生成对比报告
bash scripts/evaluate_rank_comparison.sh

# 查看报告
cat evaluation/rank_comparison_report.md
```

---

## 文档结构

```
llm_finetune/
├── README.md              # 项目概述（本文件）
├── GUIDE.md              # 详细操作流程指南
├── TUTORIAL.md           # LoRA 和 LlamaFactory 教程
├── .gitignore            # Git 忽略文件
│
├── configs/              # 训练配置（3个rank对比实验）
│   ├── qwen2_7b_lora_rank32.yaml
│   ├── qwen2_7b_lora_sft.yaml       # rank64（推荐）
│   ├── qwen2_7b_lora_rank128.yaml
│   └── dataset_info.json
│
├── scripts/              # 核心脚本
│   ├── prepare_data.py                # 数据预处理
│   ├── setup_server.sh                # 服务器配置
│   ├── prepare_training.sh            # 训练准备
│   ├── run_rank_comparison.sh         # 训练3个实验
│   ├── evaluate_rank_comparison.sh    # 评估3个实验
│   ├── generate_report.py           # 自动生成报告
│   ├── inference.py                 # 推理
│   └── evaluate.py                  # 评估
│
├── data/                 # 训练数据（运行后生成）
├── evaluation/           # 评估结果（运行后生成）
└── app/                  # Streamlit 应用
    └── main.py
```

---

## 核心文档

### 📖 必读

1. **[GUIDE.md](GUIDE.md)** - 完整操作流程 + 实验规划
   - 从零到一：本地准备数据 → 服务器配置 → 训练 → 评估 → 下载结果
   - 完整实验设计（LoRA + 超参数，共 8 组实验）
   - 运行策略（分阶段 vs 一次性）
   - 实验记录表与预期结果

2. **[TUTORIAL.md](TUTORIAL.md)** - LoRA 和 LlamaFactory 教程
   - LoRA 参数详解（rank, alpha, dropout, target）
   - 训练超参数（lr, epochs, batch size）
   - LlamaFactory 使用方法
   - 参数调优实践

---

## 硬件配置

| 资源 | 配置 |
|------|------|
| GPU | NVIDIA A800 (80GB) |
| 推荐模型 | **Qwen2-7B-Instruct** |
| 训练显存 | ~24GB (LoRA rank=64) |
| 训练时间 | ~4-6 小时 / 实验 |

> 💡 A800 80GB 完全可以支持 7B 甚至 14B 模型，**1.5B 过于保守**

---

## 推荐配置

### LoRA 参数

```yaml
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.1
lora_target: all
```

### 训练参数

```yaml
learning_rate: 2.0e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
bf16: true
flash_attn: fa2
```

---

## 预期效果

| 指标 | 目标值 | 说明 |
|------|--------|------|
| ROUGE-L F1 | >0.40 | 生成质量 |
| Priority 准确率 | >85% | 优先级分类 |
| Type 准确率 | >80% | 工单类型分类 |
| 人工评估总分 | >4.0/5.0 | 整体满意度 |

---

## 对比实验

### 推荐：LoRA Rank 对比（3个实验）

**使用经验值超参**，只对比不同 rank 的效果：

```bash
# 一键运行3个实验（约 6-7.5 小时）
bash scripts/run_rank_comparison.sh
```

**实验配置**：
- Rank 32: 快速（~2h）
- Rank 64: 推荐（~2.5h）⭐
- Rank 128: 慢速（~3h）

**固定超参**（经验值）：
- Learning Rate: 2e-4
- Epochs: 3
- Warmup: 0.05

**评估**：
```bash
bash scripts/evaluate_rank_comparison.sh
```

**输出**：
- `evaluation/rank_comparison_report.md` - LoRA Rank 对比报告

---

## 常用命令

```bash
# 数据准备
python scripts/prepare_data.py --task_type multi_task

# 训练
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml

# Web UI
llamafactory-cli webui

# 评估
python scripts/inference.py --model_path ... --test_data ...
python scripts/evaluate.py --predictions ... --references ...

# 批量实验
bash scripts/run_experiments.sh
bash scripts/evaluate_all.sh
```

---

## License

MIT
