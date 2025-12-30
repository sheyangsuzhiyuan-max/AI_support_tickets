#!/bin/bash
# 对比 Base Model vs Fine-tuned Model
# 目的：验证 fine-tuning 是否真的带来提升

set -e

echo "=========================================="
echo "Base Model vs Fine-tuned Model Comparison"
echo "=========================================="

# 配置
BASE_MODEL="Qwen/Qwen2-7B-Instruct"  # 修改为你的 base model 路径
FINETUNED_MODEL="/mnt/kai_ckp/alex/LLaMA-Factory/saves/qwen2-7b-ticket-lora-rank128/checkpoint-xxx"  # 修改为你的 fine-tuned model 路径
TEST_DATA="./data/alpaca_multi_task_test.json"
OUTPUT_DIR="./evaluation/base_vs_finetuned"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Step 1: Running inference with Base Model (no fine-tuning)..."
echo "------------------------------------------------------"
python scripts/inference.py \
    --model_path "$BASE_MODEL" \
    --test_data "$TEST_DATA" \
    --output_file "$OUTPUT_DIR/base_model_predictions.json" \
    --use_lora false \
    --max_samples 100  # 先测试 100 条，确认可行后再跑全部

echo ""
echo "Step 2: Running inference with Fine-tuned Model..."
echo "------------------------------------------------------"
python scripts/inference.py \
    --model_path "$FINETUNED_MODEL" \
    --base_model "$BASE_MODEL" \
    --test_data "$TEST_DATA" \
    --output_file "$OUTPUT_DIR/finetuned_model_predictions.json" \
    --use_lora true \
    --max_samples 100

echo ""
echo "Step 3: Evaluating Base Model..."
echo "------------------------------------------------------"
python scripts/evaluate.py \
    --predictions "$OUTPUT_DIR/base_model_predictions.json" \
    --output_dir "$OUTPUT_DIR/base_model_eval"

echo ""
echo "Step 4: Evaluating Fine-tuned Model..."
echo "------------------------------------------------------"
python scripts/evaluate.py \
    --predictions "$OUTPUT_DIR/finetuned_model_predictions.json" \
    --output_dir "$OUTPUT_DIR/finetuned_model_eval"

echo ""
echo "Step 5: Generating Comparison Report..."
echo "------------------------------------------------------"
python scripts/generate_comparison_report.py \
    --base_results "$OUTPUT_DIR/base_model_eval/results.json" \
    --finetuned_results "$OUTPUT_DIR/finetuned_model_eval/results.json" \
    --output "$OUTPUT_DIR/comparison_report.md"

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "Report saved to: $OUTPUT_DIR/comparison_report.md"
echo "=========================================="
echo ""
echo "Quick summary:"
cat "$OUTPUT_DIR/comparison_report.md" | head -30
