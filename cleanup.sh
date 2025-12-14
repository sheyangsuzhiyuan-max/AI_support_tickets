#!/bin/bash
# 清理本地模型文件和临时文件

echo "========================================="
echo "清理本地文件"
echo "========================================="

# 显示将要删除的文件大小
echo -e "\n将要删除的文件:"
echo "1. 本地BERT模型文件夹 (~512MB)"
du -sh src/model/distilbert-base-uncased notebooks/distilbert-base-uncased 2>/dev/null

echo -e "\n2. 临时生成的CSV文件"
ls -lh data/val_*.csv 2>/dev/null

echo -e "\n3. 旧的notebook备份 (如果存在)"
ls -lh notebooks/*_backup*.ipynb 2>/dev/null

echo -e "\n========================================="
read -p "确认删除以上文件? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "取消清理"
    exit 1
fi

echo -e "\n开始清理..."

# 删除本地BERT模型文件夹
if [ -d "src/model/distilbert-base-uncased" ]; then
    echo "✓ 删除 src/model/distilbert-base-uncased/"
    rm -rf src/model/distilbert-base-uncased
fi

if [ -d "notebooks/distilbert-base-uncased" ]; then
    echo "✓ 删除 notebooks/distilbert-base-uncased/"
    rm -rf notebooks/distilbert-base-uncased
fi

# 删除临时CSV文件
if ls data/val_*.csv 1> /dev/null 2>&1; then
    echo "✓ 删除 data/val_*.csv"
    rm -f data/val_*.csv
fi

# 删除notebook备份文件
if ls notebooks/*_backup*.ipynb 1> /dev/null 2>&1; then
    echo "✓ 删除 notebooks/*_backup*.ipynb"
    rm -f notebooks/*_backup*.ipynb
fi

# 删除Python缓存
echo "✓ 删除 Python 缓存文件"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 删除Jupyter notebook checkpoints
if [ -d "notebooks/.ipynb_checkpoints" ]; then
    echo "✓ 删除 Jupyter checkpoints"
    rm -rf notebooks/.ipynb_checkpoints
fi

echo -e "\n========================================="
echo "清理完成!"
echo "========================================="

# 显示清理后的空间
echo -e "\n当前项目大小:"
du -sh .

echo -e "\n保留的模型文件:"
ls -lh src/model/*.pt src/model/*.joblib 2>/dev/null

echo -e "\n✓ 本地BERT模型已删除，之后将从HuggingFace在线下载"
