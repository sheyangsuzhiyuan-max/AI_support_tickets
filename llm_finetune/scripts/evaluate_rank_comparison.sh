#!/bin/bash
# 评估 LoRA Rank 对比实验

set -e

echo "========================================="
echo "评估 LoRA Rank 对比实验"
echo "========================================="
echo ""

# 自动检测路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_DIR=$(echo $PROJECT_ROOT | sed 's|/000_ai_support_tickets.*||')

# 进入项目目录
cd $PROJECT_ROOT

# 激活环境
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate llm_finetune

# 模型列表
models=(
    "qwen2-7b-ticket-lora-rank32"
    "qwen2-7b-ticket-lora"              # rank64
    "qwen2-7b-ticket-lora-rank128"
)

# 创建评估目录
mkdir -p evaluation/rank_comparison

echo "开始评估各个模型..."
echo ""

# 评估每个模型
for model in "${models[@]}"; do
    model_path=$WORK_DIR/LLaMA-Factory/outputs/$model

    # 检查模型是否存在
    if [ ! -d "$model_path" ]; then
        echo "⚠️  模型不存在，跳过: $model"
        continue
    fi

    echo "────────────────────────────────────────"
    echo "正在评估: $model"
    echo "────────────────────────────────────────"

    # 生成预测
    python scripts/inference.py \
        --model_path $model_path \
        --base_model $WORK_DIR/models/qwen/Qwen2-7B-Instruct \
        --use_lora \
        --test_data ./data/alpaca_multi_task_test.json \
        --output ./evaluation/rank_comparison/${model}_predictions.json \
        --temperature 0.7

    # 运行评估
    python scripts/evaluate.py \
        --predictions ./evaluation/rank_comparison/${model}_predictions.json \
        --references ./data/alpaca_multi_task_test.json \
        --task_type multi_task \
        --output_dir ./evaluation/rank_comparison/$model

    echo "✓ 完成: $model"
    echo ""
done

# 生成对比报告
echo ""
echo "========================================="
echo "生成对比报告"
echo "========================================="

python scripts/generate_report.py \
    --eval_dir ./evaluation/rank_comparison \
    --output ./evaluation/rank_comparison_report.md

echo ""
echo "✓ 对比报告已生成: ./evaluation/rank_comparison_report.md"
echo ""
echo "========================================="
echo "评估完成！"
echo "========================================="
echo ""

# 显示报告预览
echo "报告预览："
echo "────────────────────────────────────────"
head -n 40 ./evaluation/rank_comparison_report.md
echo "────────────────────────────────────────"
echo ""
echo "查看完整报告："
echo "  cat ./evaluation/rank_comparison_report.md"
