# 操作指南

## 📁 目录结构说明

本项目使用以下目录结构（脚本会自动检测）：

```
/mnt/kai_ckp/alex/                      # 你的工作目录
├── 000_ai_support_tickets/             # Git 克隆的项目
│   ├── llm_finetune/                   # 微调子项目
│   └── data/                           # 原始数据
├── models/                             # 模型目录
│   └── qwen/Qwen2-7B-Instruct/
└── LLaMA-Factory/                      # 训练框架
    └── outputs/                        # 训练输出
```

**注意**：文档中的示例路径基于上述结构，实际使用时脚本会自动检测你的工作目录。

---

## 🎯 快速开始

### LoRA Rank 对比实验（3个）

使用经验值超参，只对比不同 rank 的效果：

```bash
# SSH 登录服务器
ssh username@server
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune

# 前台运行（可以看到实时输出）
bash scripts/run_rank_comparison.sh

# 如果想推后台并断开SSH：
# 1. 按 Ctrl+Z 暂停
# 2. 输入 bg 让它后台继续运行
# 3. 输入 disown 脱离终端
# 4. 可以安全退出 SSH

# 查看日志（如果已推后台）
tail -f training.log
```

**3个实验**：
- Rank 32: 快速（~2h）
- Rank 64: 推荐，平衡（~2.5h）⭐
- Rank 128: 慢速（~3h）

**固定超参**（经验值）：
- Learning Rate: 2e-4
- Epochs: 3
- Warmup: 0.05
- Batch Size: 4 × 8 = 32

**评估**：
```bash
# 训练完成后评估
bash scripts/evaluate_rank_comparison.sh
# 生成报告：evaluation/rank_comparison_report.md
```

---

## 📋 完整操作流程

### 流程图

```
本地：准备数据 → Git 上传
  ↓
服务器：配置环境 → 下载模型 → 训练 → 评估
  ↓
本地：下载结果 → 分析报告
```

---

## 第一步：本地准备数据（5分钟）

```bash
cd llm_finetune/scripts
python prepare_data.py --task_type multi_task
```

**输出**：
- `data/alpaca_multi_task_train.json` (19,782 条)
- `data/alpaca_multi_task_val.json` (4,239 条)
- `data/alpaca_multi_task_test.json` (4,240 条)

---

## 第二步：上传到服务器（Git）

```bash
# 本地：初始化 Git（首次）
git init
git add .
git commit -m "Initial commit: LLM finetune project"
git remote add origin https://your-repo.git
git push -u origin main

# 后续更新
git add .
git commit -m "Update configs"
git push
```

---

## 第三步：服务器配置（30分钟）

```bash
# SSH 登录
ssh username@server

# 克隆整个项目到你的工作目录
cd /mnt/kai_ckp/alex  # 你的工作目录
git clone https://your-repo.git

# 进入微调项目目录
cd 000_ai_support_tickets/llm_finetune

# 一键配置环境（会自动检测工作目录）
bash scripts/setup_server.sh
# 这会自动：
# - 安装 Miniconda
# - 创建 Python 环境
# - 安装 PyTorch + LlamaFactory
# - 在工作目录创建 models/ 和 LLaMA-Factory/
```

---

## 第四步：下载模型（20分钟）

```bash
# setup_server.sh 会提示是否下载
# 如果跳过了，手动下载：
cd /mnt/kai_ckp/alex  # 回到工作目录
pip install modelscope

# 下载 Qwen2-7B-Instruct (~14GB)
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./models')
"
```

**模型位置**：`/mnt/kai_ckp/alex/models/qwen/Qwen2-7B-Instruct/`

---

## 第五步：准备训练（2分钟）

```bash
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune
bash scripts/prepare_training.sh
```

**这个脚本会**：
1. 复制配置文件到 `LLaMA-Factory/configs/`
2. 链接数据文件到 `LLaMA-Factory/data/`
3. 更新配置中的模型路径

---

## 第六步：开始训练

### 方式1: 批量运行3个实验（推荐）

**方式A：前台运行**（推荐，可看到实时输出）
```bash
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune
bash scripts/run_rank_comparison.sh

# 如果中途想推后台（可选）：
# 1. 按 Ctrl+Z 暂停
# 2. 输入 bg 回车
# 3. 输入 disown 回车
# 4. 程序继续在后台运行，可以退出 SSH
```

**方式B：直接后台运行**（一开始就后台）
```bash
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune
nohup bash scripts/run_rank_comparison.sh > training.log 2>&1 &

# 查看日志
tail -f training.log              # 实时查看
tail -n 100 training.log          # 查看最后100行
grep "完成" training.log           # 搜索完成状态

# 查看进程
ps aux | grep run_rank_comparison

# 停止训练
pkill -f run_rank_comparison
```

### 方式2: 单个实验

**前台运行**：
```bash
cd /mnt/kai_ckp/alex/LLaMA-Factory
conda activate llm_finetune
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml  # rank64

# 需要时推后台: Ctrl+Z, bg, disown
```

**后台运行**：
```bash
cd /mnt/kai_ckp/alex/LLaMA-Factory
conda activate llm_finetune
nohup llamafactory-cli train configs/qwen2_7b_lora_sft.yaml > training.log 2>&1 &
tail -f training.log
```

### 方式3: Web UI（可视化）

```bash
cd /mnt/kai_ckp/alex/LLaMA-Factory
llamafactory-cli webui

# 端口转发（本地）
ssh -L 7860:localhost:7860 username@server

# 访问 http://localhost:7860
```

---

## 第七步：监控训练

**前台运行时**：
```bash
# 实时输出会直接显示在终端
# 按 Ctrl+C 可以中断训练

# 在另一个SSH窗口查看GPU
nvidia-smi
watch -n 1 nvidia-smi  # 每秒刷新
```

**后台运行时**：
```bash
# 查看进程
ps aux | grep llamafactory

# 查看日志
tail -f training.log              # 实时查看
tail -n 100 training.log          # 最后100行

# 查看 GPU
nvidia-smi
```

**使用 TensorBoard**（可选）：
```bash
# 在服务器上启动（新终端或用 nohup）
cd /mnt/kai_ckp/alex/LLaMA-Factory
nohup tensorboard --logdir ./outputs --port 6006 > tensorboard.log 2>&1 &

# 本地访问（端口转发）
ssh -L 6006:localhost:6006 username@server
# 浏览器打开: http://localhost:6006
```

---

## 第八步：评估模型

### 批量评估（推荐）

```bash
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune
bash scripts/evaluate_rank_comparison.sh
# 生成报告：evaluation/rank_comparison_report.md
```

### 单个模型评估

```bash
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune

# 生成预测
python scripts/inference.py \
    --model_path /mnt/kai_ckp/alex/LLaMA-Factory/outputs/qwen2-7b-ticket-lora \
    --base_model /mnt/kai_ckp/alex/models/qwen/Qwen2-7B-Instruct \
    --use_lora \
    --test_data ./data/alpaca_multi_task_test.json \
    --output ./evaluation/predictions.json

# 运行评估
python scripts/evaluate.py \
    --predictions ./evaluation/predictions.json \
    --references ./data/alpaca_multi_task_test.json \
    --task_type multi_task \
    --output_dir ./evaluation

# 查看报告
cat ./evaluation/evaluation_report.txt
```

---

## 第九步：下载结果

```bash
# 在服务器上压缩
cd /mnt/kai_ckp/alex/LLaMA-Factory/outputs
tar -czf qwen2-7b-ticket-lora.tar.gz qwen2-7b-ticket-lora/

# 本地下载（LoRA 权重 ~300-800 MB）
scp username@server:/mnt/kai_ckp/alex/LLaMA-Factory/outputs/qwen2-7b-ticket-lora.tar.gz ./models/

# 下载评估报告
scp username@server:/mnt/kai_ckp/alex/llm_finetune/evaluation/rank_comparison_report.md ./
```

---

## 第十步：本地测试（可选）

```bash
# 本地下载基础模型
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./models')
"

# 解压 LoRA 权重
cd models
tar -xzf qwen2-7b-ticket-lora.tar.gz

# 交互式推理
cd ..
python scripts/inference.py \
    --model_path ./models/qwen2-7b-ticket-lora \
    --base_model ./models/qwen/Qwen2-7B-Instruct \
    --use_lora \
    --interactive
```

---

## 常见问题

### Q: 训练中断了怎么办？

```bash
# 从检查点恢复
llamafactory-cli train configs/qwen2_7b_lora_sft.yaml \
    --resume_from_checkpoint ./outputs/qwen2-7b-ticket-lora/checkpoint-500
```

### Q: 显存不足？

```bash
# 减小 batch size
nano configs/qwen2_7b_lora_sft.yaml
# 修改：
# per_device_train_batch_size: 2  # 从 4 改为 2
# gradient_accumulation_steps: 16  # 从 8 改为 16
```

### Q: 如何调整参数？

参见 [TUTORIAL.md](TUTORIAL.md) 的参数调优部分。

### Q: 如何对比多个实验？

```bash
# 使用 TensorBoard（后台运行）
cd /mnt/kai_ckp/alex/LLaMA-Factory
nohup tensorboard --logdir ./outputs --port 6006 > tensorboard.log 2>&1 &
# 本地端口转发: ssh -L 6006:localhost:6006 username@server

# 或查看自动生成的对比报告
cat ./evaluation/rank_comparison_report.md
```

### Q: 前台运行如何推到后台？

```bash
# 运行中的程序：
# 1. 按 Ctrl+Z（暂停）
# 2. 输入 bg（后台继续）
# 3. 输入 disown（脱离终端）

# 查看后台进程
jobs                              # 当前终端的后台任务
ps aux | grep run_rank_comparison # 所有相关进程

# 停止程序
pkill -f run_rank_comparison
```

---

## 完整命令速查

```bash
# === 本地 ===
python prepare_data.py --task_type multi_task
git add . && git commit -m "update" && git push

# === 服务器 ===
# 首次配置
cd /mnt/kai_ckp/alex
git clone https://your-repo.git
bash scripts/setup_server.sh
bash scripts/prepare_training.sh

# 训练（3个实验）- 前台运行
cd /mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune
bash scripts/run_rank_comparison.sh
# 需要推后台: Ctrl+Z, bg, disown

# 评估
bash scripts/evaluate_rank_comparison.sh

# === 本地 ===
# 下载结果
scp username@server:/mnt/kai_ckp/alex/llm_finetune/evaluation/rank_comparison_report.md ./
```

---

## 项目文件说明

```
llm_finetune/
├── README.md              # 项目概述
├── GUIDE.md              # 👈 本文件（操作流程）
├── TUTORIAL.md           # LoRA 和 LlamaFactory 教学
│
├── configs/              # 训练配置
│   ├── qwen2_7b_lora_rank32.yaml
│   ├── qwen2_7b_lora_sft.yaml        # rank64（推荐）
│   ├── qwen2_7b_lora_rank128.yaml
│   └── dataset_info.json
│
├── scripts/              # 脚本
│   ├── prepare_data.py                # 数据预处理
│   ├── setup_server.sh                # 服务器配置
│   ├── prepare_training.sh            # 训练准备
│   ├── run_rank_comparison.sh         # 训练3个实验
│   ├── evaluate_rank_comparison.sh    # 评估3个实验
│   ├── inference.py                   # 推理
│   ├── evaluate.py                    # 评估
│   └── generate_report.py             # 生成对比报告
│
├── data/                 # 训练数据（运行后生成）
└── evaluation/           # 评估结果（运行后生成）
```
