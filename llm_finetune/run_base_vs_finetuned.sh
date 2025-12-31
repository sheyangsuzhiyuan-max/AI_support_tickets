#!/bin/bash
# 一键对比 Base Model vs Fine-tuned Model
# 使用方法: bash run_base_vs_finetuned.sh

set -e

# 不需要联网，直接用本地模型
echo "=========================================="
echo "Base vs Fine-tuned 对比实验"
echo "=========================================="

# 配置
BASE_MODEL="/mnt/kai_ckp/alex/models/qwen/Qwen2-7B-Instruct"  # 原始 Qwen base model
FINETUNED_CHECKPOINT="/mnt/kai_ckp/alex/LLaMA-Factory/outputs/qwen2-7b-ticket-lora-rank128/checkpoint-1764"
TEST_DATA="/mnt/kai_ckp/alex/AI_support_tickets/llm_finetune/data/alpaca_multi_task_test.json"
MAX_SAMPLES=100  # 先测100条，成功后改为4240

# 输出目录
OUT_DIR="./evaluation/base_vs_finetuned"
mkdir -p "$OUT_DIR"

# 已有的 finetuned 评估结果
FINETUNED_EVAL="./evaluation/rank_comparison/qwen2-7b-ticket-lora-rank128/evaluation_results.json"

# 跳过 inference 步骤（假设 base_predictions.json 已生成）

echo ""
echo "3️⃣  评估 Base Model"
echo "----------------------------------------"
python scripts/evaluate.py \
    --predictions "$OUT_DIR/base_predictions.json" \
    --references "$TEST_DATA" \
    --output_dir "$OUT_DIR/base_eval"

echo ""
echo "4️⃣  使用已有的 Fine-tuned Model 评估结果"
echo "----------------------------------------"
echo "复制已有结果: $FINETUNED_EVAL"
mkdir -p "$OUT_DIR/finetuned_eval"
cp "$FINETUNED_EVAL" "$OUT_DIR/finetuned_eval/results.json"

echo ""
echo "5️⃣  生成对比报告"
echo "----------------------------------------"
python scripts/generate_comparison_report.py \
    --base_results "$OUT_DIR/base_eval/evaluation_results.json" \
    --finetuned_results "$OUT_DIR/finetuned_eval/results.json" \
    --output "$OUT_DIR/comparison_report.md"

echo ""
echo "✅ 完成！查看结果："
echo "   $OUT_DIR/comparison_report.md"
echo ""
cat "$OUT_DIR/comparison_report.md" | head -40
