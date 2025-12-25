#!/usr/bin/env python3
"""
数据预处理脚本 - 将工单数据转换为 LlamaFactory 所需的格式

支持两种任务模式:
1. 回复生成 (response_generation): subject + body -> answer
2. 多任务 (multi_task): subject + body -> type + queue + priority + answer

输出格式: Alpaca format (LlamaFactory 标准格式)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import re


def clean_text(text: str) -> str:
    """清理文本，保留必要信息"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # 移除多余空白，但保留段落结构
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def format_tags(row: pd.Series) -> str:
    """格式化标签字段"""
    tags = []
    for i in range(1, 9):
        tag = row.get(f'tag_{i}')
        if pd.notna(tag) and tag:
            tags.append(str(tag))
    return ', '.join(tags) if tags else 'None'


def create_response_generation_sample(row: pd.Series) -> Dict:
    """
    创建回复生成任务的样本

    Alpaca 格式:
    {
        "instruction": "任务指令",
        "input": "用户输入",
        "output": "期望输出"
    }
    """
    subject = clean_text(row.get('subject', ''))
    body = clean_text(row.get('body', ''))
    answer = clean_text(row.get('answer', ''))

    # 构建输入
    if subject:
        user_input = f"Subject: {subject}\n\nContent:\n{body}"
    else:
        user_input = body

    instruction = """You are a professional customer support agent. Based on the customer's ticket below, generate a helpful, professional, and empathetic response.

Guidelines:
- Address the customer's specific concerns
- Provide clear and actionable solutions
- Maintain a professional yet friendly tone
- Use placeholders like <name>, <tel_num>, <email> for personal information"""

    return {
        "instruction": instruction,
        "input": user_input,
        "output": answer
    }


def create_multi_task_sample(row: pd.Series) -> Dict:
    """
    创建多任务样本（分类 + 生成）

    输出格式:
    Classification:
    - Type: {type}
    - Queue: {queue}
    - Priority: {priority}

    Response:
    {answer}
    """
    subject = clean_text(row.get('subject', ''))
    body = clean_text(row.get('body', ''))
    answer = clean_text(row.get('answer', ''))
    ticket_type = row.get('type', 'Unknown')
    queue = row.get('queue', 'Unknown')
    priority = row.get('priority', 'medium')
    tags = format_tags(row)

    # 构建输入
    if subject:
        user_input = f"Subject: {subject}\n\nContent:\n{body}"
    else:
        user_input = body

    instruction = """You are an intelligent customer support system. Analyze the ticket below and:
1. Classify it by type, queue, and priority
2. Generate a professional response

Available classifications:
- Type: Incident, Request, Problem, Change
- Queue: Technical Support, Product Support, Customer Service, IT Support, Billing and Payments, General Inquiry, Returns and Exchanges, Sales
- Priority: high, medium, low"""

    output = f"""Classification:
- Type: {ticket_type}
- Queue: {queue}
- Priority: {priority}
- Tags: {tags}

Response:
{answer}"""

    return {
        "instruction": instruction,
        "input": user_input,
        "output": output
    }


def create_classification_only_sample(row: pd.Series) -> Dict:
    """
    创建纯分类任务样本
    """
    subject = clean_text(row.get('subject', ''))
    body = clean_text(row.get('body', ''))
    ticket_type = row.get('type', 'Unknown')
    queue = row.get('queue', 'Unknown')
    priority = row.get('priority', 'medium')

    if subject:
        user_input = f"Subject: {subject}\n\nContent:\n{body}"
    else:
        user_input = body

    instruction = """Classify this customer support ticket into the following categories:
- Type: Incident, Request, Problem, Change
- Queue: Technical Support, Product Support, Customer Service, IT Support, Billing and Payments, General Inquiry, Returns and Exchanges, Sales
- Priority: high, medium, low

Output format: Type: X | Queue: Y | Priority: Z"""

    output = f"Type: {ticket_type} | Queue: {queue} | Priority: {priority}"

    return {
        "instruction": instruction,
        "input": user_input,
        "output": output
    }


def process_dataset(
    input_path: Path,
    output_path: Path,
    task_type: str = "response_generation",
    max_samples: Optional[int] = None
) -> int:
    """
    处理数据集

    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 JSON 文件路径
        task_type: 任务类型 (response_generation, multi_task, classification)
        max_samples: 最大样本数（用于测试）

    Returns:
        处理的样本数
    """
    # 读取数据
    df = pd.read_csv(input_path)

    if max_samples:
        df = df.head(max_samples)

    # 选择转换函数
    if task_type == "response_generation":
        convert_func = create_response_generation_sample
    elif task_type == "multi_task":
        convert_func = create_multi_task_sample
    elif task_type == "classification":
        convert_func = create_classification_only_sample
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # 转换数据
    samples = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            sample = convert_func(row)
            # 验证样本有效性
            if sample['input'] and sample['output']:
                samples.append(sample)
            else:
                skipped += 1
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to error: {e}")
            skipped += 1

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(samples)} samples, skipped {skipped}")
    return len(samples)


def create_dataset_info(output_dir: Path, task_type: str):
    """
    创建 LlamaFactory 所需的 dataset_info.json
    """
    dataset_info = {
        f"ticket_{task_type}_train": {
            "file_name": f"alpaca_{task_type}_train.json",
            "formatting": "alpaca"
        },
        f"ticket_{task_type}_val": {
            "file_name": f"alpaca_{task_type}_val.json",
            "formatting": "alpaca"
        },
        f"ticket_{task_type}_test": {
            "file_name": f"alpaca_{task_type}_test.json",
            "formatting": "alpaca"
        }
    }

    output_path = output_dir / "dataset_info.json"

    # 如果已存在，合并
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
        existing.update(dataset_info)
        dataset_info = existing

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset info saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for LlamaFactory')
    parser.add_argument('--data_dir', type=str,
                        default='../../data/processed',
                        help='Directory containing train/val/test CSV files')
    parser.add_argument('--output_dir', type=str,
                        default='../data',
                        help='Output directory for JSON files')
    parser.add_argument('--task_type', type=str,
                        choices=['response_generation', 'multi_task', 'classification'],
                        default='multi_task',
                        help='Task type for data formatting')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples per split (for testing)')

    args = parser.parse_args()

    # 获取项目根目录
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    print(f"Task type: {args.task_type}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # 处理每个数据集
    splits = ['train', 'val', 'test']

    for split in splits:
        input_path = data_dir / f"tickets_{split}.csv"
        output_path = output_dir / f"alpaca_{args.task_type}_{split}.json"

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping...")
            continue

        print(f"\nProcessing {split}...")
        n_samples = process_dataset(
            input_path=input_path,
            output_path=output_path,
            task_type=args.task_type,
            max_samples=args.max_samples
        )
        print(f"Saved to {output_path}")

    # 创建 dataset_info.json
    create_dataset_info(output_dir, args.task_type)

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
