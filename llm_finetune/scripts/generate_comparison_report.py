#!/usr/bin/env python3
"""
Generate Base Model vs Fine-tuned Model Comparison Report
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_results(filepath):
    """Load evaluation results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract aggregated field if present
    if 'aggregated' in data:
        return data['aggregated']
    return data


def calculate_improvement(base_val, ft_val):
    """Calculate improvement percentage"""
    if base_val == 0:
        return "N/A"
    improvement = ((ft_val - base_val) / base_val) * 100
    return f"{improvement:+.2f}%"


def generate_report(base_results, finetuned_results, output_path):
    """Generate comparison report"""

    report = []
    report.append("# Base Model vs Fine-tuned Model Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")

    # 1. Executive Summary
    report.append("## Executive Summary\n")

    # Classification metrics
    type_base = base_results.get('type_accuracy', 0)
    type_ft = finetuned_results.get('type_accuracy', 0)
    type_improvement = calculate_improvement(type_base, type_ft)

    priority_base = base_results.get('priority_accuracy', 0)
    priority_ft = finetuned_results.get('priority_accuracy', 0)
    priority_improvement = calculate_improvement(priority_base, priority_ft)

    queue_base = base_results.get('queue_accuracy', 0)
    queue_ft = finetuned_results.get('queue_accuracy', 0)
    queue_improvement = calculate_improvement(queue_base, queue_ft)

    report.append(f"- **Ticket Type Classification:** {type_base*100:.2f}% → {type_ft*100:.2f}% ({type_improvement})")
    report.append(f"- **Priority Classification:** {priority_base*100:.2f}% → {priority_ft*100:.2f}% ({priority_improvement})")
    report.append(f"- **Queue Routing:** {queue_base*100:.2f}% → {queue_ft*100:.2f}% ({queue_improvement})\n")

    # ROUGE scores
    rouge1_base = base_results.get('avg_rouge1_f', 0)
    rouge1_ft = finetuned_results.get('avg_rouge1_f', 0)
    rougeL_base = base_results.get('avg_rougeL_f', 0)
    rougeL_ft = finetuned_results.get('avg_rougeL_f', 0)

    report.append(f"- **Response Quality (ROUGE-1):** {rouge1_base:.4f} → {rouge1_ft:.4f} ({calculate_improvement(rouge1_base, rouge1_ft)})")
    report.append(f"- **Response Quality (ROUGE-L):** {rougeL_base:.4f} → {rougeL_ft:.4f} ({calculate_improvement(rougeL_base, rougeL_ft)})\n")

    # 2. Detailed Metrics
    report.append("\n## Detailed Comparison\n")
    report.append("### Classification Accuracy\n")
    report.append("| Metric | Base Model | Fine-tuned | Improvement |")
    report.append("|--------|-----------|------------|-------------|")
    report.append(f"| Type Accuracy | {type_base*100:.2f}% | {type_ft*100:.2f}% | {type_improvement} |")
    report.append(f"| Priority Accuracy | {priority_base*100:.2f}% | {priority_ft*100:.2f}% | {priority_improvement} |")
    report.append(f"| Queue Accuracy | {queue_base*100:.2f}% | {queue_ft*100:.2f}% | {queue_improvement} |\n")

    report.append("### Response Generation Quality\n")
    report.append("| Metric | Base Model | Fine-tuned | Improvement |")
    report.append("|--------|-----------|------------|-------------|")
    report.append(f"| ROUGE-1 F1 | {rouge1_base:.4f} | {rouge1_ft:.4f} | {calculate_improvement(rouge1_base, rouge1_ft)} |")
    report.append(f"| ROUGE-2 F1 | {base_results.get('avg_rouge2_f', 0):.4f} | {finetuned_results.get('avg_rouge2_f', 0):.4f} | {calculate_improvement(base_results.get('avg_rouge2_f', 0), finetuned_results.get('avg_rouge2_f', 0))} |")
    report.append(f"| ROUGE-L F1 | {rougeL_base:.4f} | {rougeL_ft:.4f} | {calculate_improvement(rougeL_base, rougeL_ft)} |\n")

    # 3. Analysis
    report.append("\n## Analysis\n")

    # Determine if fine-tuning was effective
    classification_improved = (type_ft > type_base) or (priority_ft > priority_base) or (queue_ft > queue_base)
    generation_improved = (rouge1_ft > rouge1_base) or (rougeL_ft > rougeL_base)

    if classification_improved and generation_improved:
        report.append("✅ **Fine-tuning was effective!**\n")
        report.append("- Classification accuracy improved across multiple metrics")
        report.append("- Response generation quality improved")
        report.append("- Domain adaptation through LoRA successfully captured task-specific patterns\n")
    elif classification_improved:
        report.append("⚠️ **Partial improvement**\n")
        report.append("- Classification accuracy improved")
        report.append("- Response generation needs further optimization\n")
    else:
        report.append("❌ **Fine-tuning did not show significant improvement**\n")
        report.append("- Consider adjusting hyperparameters (learning rate, LoRA rank)")
        report.append("- Check training data quality and distribution\n")

    # 4. Recommendations
    report.append("\n## Recommendations\n")

    if type_ft < 0.80:
        report.append("- **Type Classification:** Consider collecting more training examples for underrepresented types")
    if priority_ft < 0.75:
        report.append("- **Priority Classification:** Review priority labeling criteria - may need clearer definitions")
    if queue_ft < 0.70:
        report.append("- **Queue Routing:** 8-way classification is challenging - consider hierarchical classification")
    if rougeL_ft < 0.30:
        report.append("- **Response Quality:** Increase training epochs or use larger LoRA rank for better generation")

    # Write report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n✅ Comparison report generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument('--base_results', type=str, required=True, help="Base model results JSON")
    parser.add_argument('--finetuned_results', type=str, required=True, help="Fine-tuned model results JSON")
    parser.add_argument('--output', type=str, required=True, help="Output report path")

    args = parser.parse_args()

    # Load results
    base_results = load_results(args.base_results)
    finetuned_results = load_results(args.finetuned_results)

    # Generate report
    generate_report(base_results, finetuned_results, args.output)


if __name__ == "__main__":
    main()
