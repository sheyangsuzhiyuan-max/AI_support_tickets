#!/bin/bash
# 一键对比 Base Model vs Fine-tuned Model
# 使用方法: bash run_base_vs_finetuned.sh

set -e

echo "=========================================="
echo "Base vs Fine-tuned 对比实验"
echo "=========================================="

# 配置
BASE_MODEL="/mnt/kai_ckp/alex/LLaMA-Factory/models/qwen/Qwen2-7B-Instruct"
FINETUNED_CHECKPOINT="/mnt/kai_ckp/alex/LLaMA-Factory/outputs/qwen2-7b-ticket-lora-rank128/checkpoint-1764"  # 使用最后一个checkpoint
TEST_DATA="/mnt/kai_ckp/alex/000_ai_support_tickets/llm_finetune/data/alpaca_multi_task_test.json"
MAX_SAMPLES=100  # 先测100条，成功后改为4240

# 输出目录
OUT_DIR="./evaluation/base_vs_finetuned"
mkdir -p "$OUT_DIR"

echo ""
echo "1️⃣  测试 Base Model（未fine-tune）"
echo "----------------------------------------"
python scripts/inference.py \
    --model_path "$BASE_MODEL" \
    --test_data "$TEST_DATA" \
    --output_file "$OUT_DIR/base_predictions.json" \
    --use_lora false \
    --max_samples $MAX_SAMPLES

echo ""
echo "2️⃣  测试 Fine-tuned Model"
echo "----------------------------------------"
python scripts/inference.py \
    --model_path "$FINETUNED_CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --test_data "$TEST_DATA" \
    --output_file "$OUT_DIR/finetuned_predictions.json" \
    --use_lora true \
    --max_samples $MAX_SAMPLES

echo ""
echo "3️⃣  评估 Base Model"
echo "----------------------------------------"
python scripts/evaluate.py \
    --predictions "$OUT_DIR/base_predictions.json" \
    --output_dir "$OUT_DIR/base_eval"

echo ""
echo "4️⃣  评估 Fine-tuned Model"
echo "----------------------------------------"
python scripts/evaluate.py \
    --predictions "$OUT_DIR/finetuned_predictions.json" \
    --output_dir "$OUT_DIR/finetuned_eval"

echo ""
echo "5️⃣  生成对比报告"
echo "----------------------------------------"
python scripts/generate_comparison_report.py \
    --base_results "$OUT_DIR/base_eval/results.json" \
    --finetuned_results "$OUT_DIR/finetuned_eval/results.json" \
    --output "$OUT_DIR/comparison_report.md"

echo ""
echo "✅ 完成！查看结果："
echo "   $OUT_DIR/comparison_report.md"
echo ""
cat "$OUT_DIR/comparison_report.md" | head -40
