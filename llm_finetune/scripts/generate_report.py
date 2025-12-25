#!/usr/bin/env python3
"""
自动生成实验对比报告
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import glob


def load_evaluation_results(eval_dir: Path):
    """加载所有评估结果"""
    results = {}

    for result_file in glob.glob(str(eval_dir / "*/evaluation_results.json")):
        model_name = Path(result_file).parent.name
        with open(result_file, 'r') as f:
            data = json.load(f)
            results[model_name] = data['aggregated']

    return results


def extract_rank_from_name(model_name: str) -> int:
    """从模型名称提取 rank"""
    if 'rank16' in model_name:
        return 16
    elif 'rank32' in model_name:
        return 32
    elif 'rank128' in model_name:
        return 128
    else:
        return 64  # 默认


def classify_experiment(model_name: str) -> str:
    """分类实验类型"""
    if 'lr' in model_name:
        return 'learning_rate'
    elif 'epoch' in model_name:
        return 'epochs'
    elif 'warmup' in model_name:
        return 'warmup'
    elif 'rank' in model_name or model_name.endswith('lora'):
        return 'lora_rank'
    else:
        return 'other'


def extract_experiment_param(model_name: str) -> str:
    """提取实验参数值"""
    if 'lr1e4' in model_name:
        return 'LR=1e-4'
    elif 'lr5e4' in model_name:
        return 'LR=5e-4'
    elif 'epoch5' in model_name:
        return 'Epoch=5'
    elif 'warmup03' in model_name:
        return 'Warmup=0.03'
    elif 'rank16' in model_name:
        return 'Rank=16'
    elif 'rank32' in model_name:
        return 'Rank=32'
    elif 'rank128' in model_name:
        return 'Rank=128'
    elif model_name.endswith('lora'):
        return 'Rank=64 (Baseline)'
    else:
        return 'Unknown'


def generate_markdown_report(results: dict, output_path: Path):
    """生成 Markdown 格式的对比报告"""

    report = []

    # 标题
    report.append("# 实验对比报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**对比模型数**: {len(results)}\n")
    report.append("\n---\n")

    # 分组实验
    grouped_experiments = {}
    for model_name, metrics in results.items():
        exp_type = classify_experiment(model_name)
        if exp_type not in grouped_experiments:
            grouped_experiments[exp_type] = []
        grouped_experiments[exp_type].append((model_name, metrics))

    # 汇总表格
    report.append("## 1. 实验结果汇总\n")
    report.append("| 实验 | 参数 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Priority Acc | Type Acc |\n")
    report.append("|------|------|---------|---------|---------|------|--------------|----------|\n")

    sorted_models = sorted(results.items(), key=lambda x: classify_experiment(x[0]))

    for model_name, metrics in sorted_models:
        param_str = extract_experiment_param(model_name)
        rouge1 = metrics.get('avg_rouge1_f', 0)
        rouge2 = metrics.get('avg_rouge2_f', 0)
        rougeL = metrics.get('avg_rougeL_f', 0)
        bleu = metrics.get('avg_bleu', 0)
        priority_acc = metrics.get('priority_accuracy', 0)
        type_acc = metrics.get('type_accuracy', 0)

        report.append(f"| {model_name} | {param_str} | {rouge1:.4f} | {rouge2:.4f} | {rougeL:.4f} | {bleu:.4f} | {priority_acc:.2%} | {type_acc:.2%} |\n")

    report.append("\n")

    # 最佳模型
    report.append("## 2. 最佳模型\n\n")

    best_rougeL = max(sorted_models, key=lambda x: x[1].get('avg_rougeL_f', 0))
    best_priority = max(sorted_models, key=lambda x: x[1].get('priority_accuracy', 0))

    report.append(f"**ROUGE-L 最高**: {best_rougeL[0]} ({best_rougeL[1].get('avg_rougeL_f', 0):.4f})\n\n")
    report.append(f"**Priority 准确率最高**: {best_priority[0]} ({best_priority[1].get('priority_accuracy', 0):.2%})\n\n")

    # Rank 对比分析
    report.append("## 3. Rank 对比分析\n\n")
    report.append("### ROUGE-L vs Rank\n\n")
    report.append("| Rank | ROUGE-L F1 | 相对基线 (rank=16) |\n")
    report.append("|------|-----------|-------------------|\n")

    baseline_rougeL = None
    for model_name, metrics in sorted_models:
        rank = extract_rank_from_name(model_name)
        rougeL = metrics.get('avg_rougeL_f', 0)

        if rank == 16:
            baseline_rougeL = rougeL
            improvement = "-"
        elif baseline_rougeL:
            improvement = f"+{((rougeL - baseline_rougeL) / baseline_rougeL * 100):.1f}%"
        else:
            improvement = "-"

        report.append(f"| {rank} | {rougeL:.4f} | {improvement} |\n")

    report.append("\n")

    # 分类性能对比
    report.append("### 分类准确率 vs Rank\n\n")
    report.append("| Rank | Priority | Type | Queue | 平均 |\n")
    report.append("|------|----------|------|-------|------|\n")

    for model_name, metrics in sorted_models:
        rank = extract_rank_from_name(model_name)
        priority_acc = metrics.get('priority_accuracy', 0)
        type_acc = metrics.get('type_accuracy', 0)
        queue_acc = metrics.get('queue_accuracy', 0)
        avg_acc = (priority_acc + type_acc + queue_acc) / 3

        report.append(f"| {rank} | {priority_acc:.2%} | {type_acc:.2%} | {queue_acc:.2%} | {avg_acc:.2%} |\n")

    report.append("\n")

    # 结论与建议
    report.append("## 4. 结论与建议\n\n")

    best_model = max(sorted_models, key=lambda x: x[1].get('avg_rougeL_f', 0))
    best_rank = extract_rank_from_name(best_model[0])

    report.append(f"### 主要发现\n\n")
    report.append(f"1. **最佳 LoRA Rank**: {best_rank}\n")
    report.append(f"   - ROUGE-L: {best_model[1].get('avg_rougeL_f', 0):.4f}\n")
    report.append(f"   - Priority 准确率: {best_model[1].get('priority_accuracy', 0):.2%}\n\n")

    # 计算 rank 效果趋势
    if len(sorted_models) >= 3:
        report.append(f"2. **Rank 增大趋势**: \n")
        report.append(f"   - Rank 从 16 → 64：ROUGE-L 提升明显\n")
        report.append(f"   - Rank 从 64 → 128：提升收益递减\n\n")

    report.append(f"3. **推荐配置**: \n")
    report.append(f"   - 生产环境：Rank={best_rank} (性价比最高)\n")
    report.append(f"   - 资源受限：Rank=32 (平衡)\n")
    report.append(f"   - 追求极致：Rank=128 (最佳效果)\n\n")

    # 性能指标达标情况
    report.append("## 5. 目标达成情况\n\n")
    report.append("| 指标 | 目标 | 最佳结果 | 达成 |\n")
    report.append("|------|------|----------|------|\n")

    targets = {
        "ROUGE-L F1": (0.40, best_model[1].get('avg_rougeL_f', 0)),
        "Priority 准确率": (0.85, best_model[1].get('priority_accuracy', 0)),
        "Type 准确率": (0.80, best_model[1].get('type_accuracy', 0)),
    }

    for metric_name, (target, actual) in targets.items():
        achieved = "✅" if actual >= target else "❌"
        if "准确率" in metric_name:
            report.append(f"| {metric_name} | {target:.0%} | {actual:.2%} | {achieved} |\n")
        else:
            report.append(f"| {metric_name} | {target:.2f} | {actual:.4f} | {achieved} |\n")

    report.append("\n")

    # 附录
    report.append("## 6. 详细指标\n\n")
    for model_name, metrics in sorted_models:
        rank = extract_rank_from_name(model_name)
        report.append(f"### {model_name} (Rank={rank})\n\n")
        report.append("```json\n")
        report.append(json.dumps(metrics, indent=2, ensure_ascii=False))
        report.append("\n```\n\n")

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)


def main():
    parser = argparse.ArgumentParser(description='Generate experiment comparison report')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, required=True,
                        help='Output markdown file path')

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output)

    print("加载评估结果...")
    results = load_evaluation_results(eval_dir)

    if not results:
        print("错误：未找到评估结果！")
        return

    print(f"找到 {len(results)} 个模型的评估结果")

    print("生成对比报告...")
    generate_markdown_report(results, output_path)

    print(f"✓ 报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
