#!/bin/bash
# BERT训练环境修复脚本

echo "================================================"
echo "BERT模型训练环境检查与修复"
echo "================================================"
echo ""

# 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python --version 2>&1)
echo "   $python_version"
echo ""

# 检查并安装transformers
echo "2. 检查transformers库..."
if python -c "import transformers" 2>/dev/null; then
    version=$(python -c "import transformers; print(transformers.__version__)")
    echo "   ✓ transformers已安装 (版本: $version)"
else
    echo "   ✗ transformers未安装"
    echo "   正在安装transformers..."
    pip install transformers
    echo "   ✓ transformers安装完成"
fi
echo ""

# 检查PyTorch和CUDA
echo "3. 检查PyTorch和CUDA..."
python -c "
import torch
print(f'   PyTorch版本: {torch.__version__}')
print(f'   CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA版本: {torch.version.cuda}')
    print(f'   GPU数量: {torch.cuda.device_count()}')
    print(f'   GPU名称: {torch.cuda.get_device_name(0)}')
else:
    print('   ⚠️  警告: 未检测到GPU，训练将在CPU上进行（速度较慢）')
"
echo ""

# 检查其他依赖
echo "4. 检查其他必要库..."
python -c "
import sys

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'torch': 'torch'
}

missing = []
for module, package in packages.items():
    try:
        __import__(module)
        print(f'   ✓ {module}')
    except ImportError:
        print(f'   ✗ {module}')
        missing.append(package)

if missing:
    print(f'   需要安装: {\" \".join(missing)}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "正在安装缺失的依赖..."
    pip install -r requirements.txt
fi

echo ""
echo "================================================"
echo "环境检查完成！"
echo "================================================"
echo ""
echo "下一步操作："
echo "1. 如果没有GPU，考虑使用Google Colab: https://colab.research.google.com"
echo "2. 如果有GPU但CUDA不可用，重新安装PyTorch CUDA版本"
echo "3. 运行训练: jupyter notebook notebooks/04_BERT_Finetune.ipynb"
echo ""
