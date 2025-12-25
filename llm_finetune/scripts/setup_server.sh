#!/bin/bash
# 服务器环境自动配置脚本
# 使用方法：bash setup_server.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "LLM 微调项目 - 服务器环境配置脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 CUDA
echo -e "${YELLOW}[1/8] 检查 CUDA 环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ CUDA 环境正常${NC}"
else
    echo -e "${RED}✗ 未检测到 NVIDIA GPU，请确认硬件配置${NC}"
    exit 1
fi

# 2. 获取当前工作目录
echo -e "\n${YELLOW}[2/8] 检查工作目录...${NC}"
WORK_DIR=$(pwd | sed 's|/\(000_\)\?[Aa][Ii]_support_tickets.*||')
echo -e "工作目录: ${WORK_DIR}"
mkdir -p ${WORK_DIR}/models
echo -e "${GREEN}✓ 目录检查完成${NC}"

# 3. 检查 Conda
echo -e "\n${YELLOW}[3/6] 检查 Conda...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ 未找到 Conda，请先安装 Conda${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conda 已安装: $(conda --version)${NC}"

# 4. 创建虚拟环境
echo -e "\n${YELLOW}[4/6] 创建 Python 环境...${NC}"
if conda env list | grep -q "llm_finetune"; then
    echo -e "${YELLOW}→ 环境已存在，跳过创建${NC}"
else
    conda create -n llm_finetune python=3.10 -y
    echo -e "${GREEN}✓ Python 环境创建完成${NC}"
fi

# 5. 激活环境并安装 PyTorch
echo -e "\n${YELLOW}[5/6] 安装 PyTorch (CUDA 12.1)...${NC}"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate llm_finetune

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo -e "${GREEN}✓ PyTorch 安装完成${NC}"

# 6. 克隆并安装 LlamaFactory
echo -e "\n${YELLOW}[6/6] 安装 LLaMA-Factory 和依赖...${NC}"
cd ${WORK_DIR}
if [ -d "LLaMA-Factory" ]; then
    echo -e "${YELLOW}→ LLaMA-Factory 已存在，更新代码${NC}"
    cd LLaMA-Factory
    git pull
else
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
fi

pip install -e ".[torch,metrics]"
llamafactory-cli version
echo -e "${GREEN}✓ LLaMA-Factory 安装完成${NC}"

# 安装其他依赖
pip install rouge-score nltk pandas scikit-learn

# Flash Attention 2（可选，如果编译失败可跳过）
echo -e "${YELLOW}→ 尝试安装 Flash Attention 2（可能需要几分钟）...${NC}"
pip install flash-attn --no-build-isolation || echo -e "${YELLOW}⚠ Flash Attention 安装失败，将使用标准 attention${NC}"

echo -e "${GREEN}✓ LLaMA-Factory 和依赖安装完成${NC}"

# 下载模型（可选）
echo -e "\n${YELLOW}准备下载 Qwen2-7B 模型...${NC}"
echo -e "${YELLOW}注意：模型约 14GB，需要较长时间${NC}"
read -p "是否现在下载模型？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ${WORK_DIR}
    pip install modelscope
    python -c "
from modelscope import snapshot_download
print('开始下载 Qwen2-7B-Instruct...')
snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./models')
print('模型下载完成！')
"
    echo -e "${GREEN}✓ 模型下载完成${NC}"
else
    echo -e "${YELLOW}→ 跳过模型下载，稍后手动执行：${NC}"
    echo -e "   cd ${WORK_DIR} && python -c \"from modelscope import snapshot_download; snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./models')\""
fi

# 完成
echo -e "\n${GREEN}========================================"
echo "✓ 服务器环境配置完成！"
echo "========================================${NC}"
echo ""
echo "工作目录: ${WORK_DIR}"
echo "项目目录: ${WORK_DIR}/000_ai_support_tickets/llm_finetune"
echo "模型目录: ${WORK_DIR}/models"
echo "训练框架: ${WORK_DIR}/LLaMA-Factory"
echo ""
echo "下一步操作："
echo "1. 配置训练文件："
echo "   cd ${WORK_DIR}/000_ai_support_tickets/llm_finetune"
echo "   bash scripts/prepare_training.sh"
echo ""
echo "2. 启动训练："
echo "   bash scripts/run_rank_comparison.sh"
echo ""
