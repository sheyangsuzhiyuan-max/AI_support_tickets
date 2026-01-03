#!/usr/bin/env python3
"""
Inference Script - Run predictions with fine-tuned model

Features:
1. Single sample inference
2. Batch inference
3. Interactive inference
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm


def load_model(model_path: str, use_lora: bool = True, base_model: str = None):
    """
    Load model

    Args:
        model_path: Path to model (merged model or LoRA adapter)
        use_lora: Whether to use LoRA adapter
        base_model: Path to base model (required when use_lora=True)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")

    if use_lora and base_model:
        # Load base model + LoRA
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        # Load merged model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()
    print("Model loaded successfully!")

    return model, tokenizer


def build_prompt(instruction: str, input_text: str, template: str = "qwen") -> str:
    """
    Build prompt

    Args:
        instruction: Task instruction
        input_text: User input
        template: Template type
    """
    if template == "qwen":
        # Qwen2 chat template
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        # Simplified version, use tokenizer.apply_chat_template in practice
        prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Alpaca template
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
    Generate response

    Args:
        model: Model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter
        top_p: Top-p sampling
        do_sample: Whether to use sampling
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

    # Decode output (only take newly generated part)
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
    Batch inference

    Args:
        model: Model
        tokenizer: Tokenizer
        data: Input data list
        output_path: Output file path
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter
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
            "reference": item.get("output", "")  # Keep reference for evaluation
        })

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")
    return results


def interactive_mode(model, tokenizer, task_type: str = "multi_task"):
    """
    Interactive inference mode
    """
    print("\n" + "=" * 50)
    print("Smart Ticket System - Interactive Inference")
    print("Type 'quit' to exit")
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
        print("\n--- New Ticket ---")
        subject = input("Subject: ").strip()
        if subject.lower() == 'quit':
            break

        body = input("Body (multi-line input, empty line to finish):\n")
        body_lines = []
        while True:
            line = input()
            if line == "":
                break
            body_lines.append(line)
        body = "\n".join(body_lines)

        if not body:
            print("Body cannot be empty")
            continue

        # Build input
        if subject:
            input_text = f"Subject: {subject}\n\nContent:\n{body}"
        else:
            input_text = body

        prompt = build_prompt(instruction, input_text)

        print("\nGenerating response...")
        response = generate_response(model, tokenizer, prompt)

        print("\n" + "=" * 50)
        print("AI Response:")
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

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        use_lora=args.use_lora,
        base_model=args.base_model
    )

    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, args.task_type)
    elif args.test_data:
        # Batch inference
        with open(args.test_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch_inference(
            model, tokenizer, data,
            Path(args.output),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
    else:
        print("Please specify --test_data or --interactive")


if __name__ == "__main__":
    main()
