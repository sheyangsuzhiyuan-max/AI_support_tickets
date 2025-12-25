#!/bin/bash
# 准备训练文件脚本（在服务器上运行）

set -e

echo "========================================"
echo "准备训练配置文件"
echo "========================================"
echo ""

# 自动检测路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_DIR=$(echo $PROJECT_ROOT | sed 's|/000_ai_support_tickets.*||')
LLAMAFACTORY_ROOT=$WORK_DIR/LLaMA-Factory
MODEL_PATH=$WORK_DIR/models/qwen/Qwen2-7B-Instruct

echo "项目路径: $PROJECT_ROOT"
echo "工作路径: $WORK_DIR"
echo ""

# 1. 复制配置文件
echo "[1/4] 复制配置文件到 LlamaFactory..."
cp $PROJECT_ROOT/configs/qwen2_7b_lora_*.yaml $LLAMAFACTORY_ROOT/configs/
cp $PROJECT_ROOT/configs/dataset_info.json $LLAMAFACTORY_ROOT/data/
echo "✓ 配置文件复制完成（3个实验配置）"

# 2. 创建数据符号链接
echo ""
echo "[2/4] 创建数据文件链接..."
cd $LLAMAFACTORY_ROOT/data
ln -sf $PROJECT_ROOT/data/alpaca_multi_task_train.json .
ln -sf $PROJECT_ROOT/data/alpaca_multi_task_val.json .
ln -sf $PROJECT_ROOT/data/alpaca_multi_task_test.json .
echo "✓ 数据链接创建完成"

# 3. 更新配置文件中的模型路径
echo ""
echo "[3/4] 更新配置文件中的模型路径..."
for config in $LLAMAFACTORY_ROOT/configs/qwen2_7b_lora_*.yaml; do
    sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_PATH|" $config
    echo "  更新: $(basename $config)"
done
echo "✓ 模型路径更新完成（3个配置）"

# 4. 验证配置
echo ""
echo "[4/4] 验证配置..."
echo "模型路径: $MODEL_PATH"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ 模型存在"
    ls -lh $MODEL_PATH | head -5
else
    echo "✗ 模型不存在！请先下载模型"
    exit 1
fi

echo ""
echo "数据文件:"
ls -lh $LLAMAFACTORY_ROOT/data/alpaca_multi_task_*.json

echo ""
echo "========================================"
echo "✓ 准备工作完成！"
echo "========================================"
echo ""
echo "现在可以启动训练："
echo ""
echo "  # 批量运行3个实验（推荐）"
echo "  cd $PROJECT_ROOT"
echo "  bash scripts/run_rank_comparison.sh"
echo ""
echo "  # 或单个实验"
echo "  cd $LLAMAFACTORY_ROOT"
echo "  llamafactory-cli train configs/qwen2_7b_lora_sft.yaml"
echo ""
