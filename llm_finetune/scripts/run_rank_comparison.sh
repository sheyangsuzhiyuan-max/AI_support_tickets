#!/bin/bash
# LoRA Rank 对比实验（3个实验）

set -e

echo "========================================="
echo "LoRA Rank 对比实验"
echo "========================================="
echo ""

# 自动检测路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_DIR=$(echo $PROJECT_ROOT | sed 's|/\(000_\)\?[Aa][Ii]_support_tickets.*||')

# 进入 LlamaFactory 目录
cd $WORK_DIR/LLaMA-Factory

# 激活环境
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate llm_finetune

# 3 个实验配置
experiments=(
    "qwen2_7b_lora_rank32"      # 小 rank，快速
    "qwen2_7b_lora_sft"         # 中 rank，平衡（推荐）
    "qwen2_7b_lora_rank128"     # 大 rank，慢但可能效果好
)

echo "实验列表:"
echo "  1. Rank 32 (快速，~1.5-2h)"
echo "  2. Rank 64 (推荐，~2-2.5h)"
echo "  3. Rank 128 (慢速，~2.5-3h)"
echo ""
echo "预计总时间: ~6-7.5 小时"
echo ""

# 运行实验
for exp in "${experiments[@]}"; do
    echo "────────────────────────────────────────"
    echo "训练: $exp"
    echo "────────────────────────────────────────"

    if [ -f "configs/${exp}.yaml" ]; then
        llamafactory-cli train configs/${exp}.yaml
        echo "✓ 完成: $exp"
    else
        echo "⚠️  配置文件不存在: configs/${exp}.yaml"
    fi
    echo ""
done

echo ""
echo "========================================="
echo "所有实验完成！"
echo "========================================="
echo ""

# 显示结果
echo "训练结果保存在："
ls -lh ./outputs/

echo ""
echo "下一步："
echo "1. 查看 TensorBoard 对比:"
echo "   tensorboard --logdir ./outputs --port 6006"
echo ""
echo "2. 运行评估:"
echo "   cd ~/projects/llm_finetune"
echo "   bash scripts/evaluate_rank_comparison.sh"
