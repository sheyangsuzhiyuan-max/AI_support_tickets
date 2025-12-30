#!/usr/bin/env python3
"""
推理脚本 - 使用微调后的模型进行预测

支持:
1. 单条推理
2. 批量推理
3. 交互式推理
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm


def load_model(model_path: str, use_lora: bool = True, base_model: str = None):
    """
    加载模型

    Args:
        model_path: 模型路径（合并后的模型或 LoRA 适配器）
        use_lora: 是否使用 LoRA 适配器
        base_model: 基础模型路径（仅当 use_lora=True 时需要）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    print(f"Loading model from {model_path}...")

    # 检测是否为本地路径，强制使用本地文件
    is_local = os.path.exists(model_path) if model_path else False
    is_base_local = os.path.exists(base_model) if base_model else False

    print(f"Model path is local: {is_local}")
    if base_model:
        print(f"Base model path is local: {is_base_local}")

    if use_lora and base_model:
        # 加载基础模型 + LoRA
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=is_base_local
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=is_base_local
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重
    else:
        # 加载合并后的模型
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=is_local
        )

    model.eval()
    print("Model loaded successfully!")

    return model, tokenizer


def build_prompt(instruction: str, input_text: str, template: str = "qwen") -> str:
    """
    构建 prompt

    Args:
        instruction: 任务指令
        input_text: 用户输入
        template: 模板类型
    """
    if template == "qwen":
        # Qwen2 chat 模板
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        # 简化版本，实际使用 tokenizer.apply_chat_template
        prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Alpaca 模板
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    生成回复

    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入 prompt
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: Top-p 采样
        do_sample: 是否采样
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码输出（只取新生成的部分）
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def batch_inference(
    model,
    tokenizer,
    data: List[Dict],
    output_path: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> List[Dict]:
    """
    批量推理

    Args:
        model: 模型
        tokenizer: 分词器
        data: 输入数据列表
        output_path: 输出文件路径
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
    """
    results = []

    for item in tqdm(data, desc="Generating responses"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        prompt = build_prompt(instruction, input_text)
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        results.append({
            "instruction": instruction,
            "input": input_text,
            "output": response,
            "reference": item.get("output", "")  # 保留参考答案用于评估
        })

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")
    return results


def interactive_mode(model, tokenizer, task_type: str = "multi_task"):
    """
    交互式推理模式
    """
    print("\n" + "=" * 50)
    print("智能工单系统 - 交互式推理")
    print("输入 'quit' 退出")
    print("=" * 50 + "\n")

    if task_type == "multi_task":
        instruction = """You are an intelligent customer support system. Analyze the ticket below and:
1. Classify it by type, queue, and priority
2. Generate a professional response

Available classifications:
- Type: Incident, Request, Problem, Change
- Queue: Technical Support, Product Support, Customer Service, IT Support, Billing and Payments, General Inquiry, Returns and Exchanges, Sales
- Priority: high, medium, low"""
    else:
        instruction = """You are a professional customer support agent. Based on the customer's ticket below, generate a helpful, professional, and empathetic response.

Guidelines:
- Address the customer's specific concerns
- Provide clear and actionable solutions
- Maintain a professional yet friendly tone
- Use placeholders like <name>, <tel_num>, <email> for personal information"""

    while True:
        print("\n--- 新工单 ---")
        subject = input("Subject: ").strip()
        if subject.lower() == 'quit':
            break

        body = input("Body (多行输入，空行结束):\n")
        body_lines = []
        while True:
            line = input()
            if line == "":
                break
            body_lines.append(line)
        body = "\n".join(body_lines)

        if not body:
            print("Body 不能为空")
            continue

        # 构建输入
        if subject:
            input_text = f"Subject: {subject}\n\nContent:\n{body}"
        else:
            input_text = body

        prompt = build_prompt(instruction, input_text)

        print("\n正在生成回复...")
        response = generate_response(model, tokenizer, prompt)

        print("\n" + "=" * 50)
        print("AI 回复:")
        print("=" * 50)
        print(response)


def main():
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--base_model', type=str, default=None,
                        help='Path to base model (if using LoRA adapter)')
    parser.add_argument('--use_lora', action='store_true',
                        help='Whether model_path is a LoRA adapter')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data JSON for batch inference')
    parser.add_argument('--output', type=str, default='./predictions.json',
                        help='Output path for predictions')
    parser.add_argument('--task_type', type=str,
                        choices=['response_generation', 'multi_task'],
                        default='multi_task')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model
    )

    if args.interactive:
        # 交互模式
        interactive_mode(model, tokenizer, args.task_type)
    elif args.test_data:
        # 批量推理
        with open(args.test_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch_inference(
            model, tokenizer, data,
            Path(args.output),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
    else:
        print("请指定 --test_data 或 --interactive")


if __name__ == "__main__":
    main()
