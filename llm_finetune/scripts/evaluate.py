#!/usr/bin/env python3
"""
评估脚本 - 支持自动评估（ROUGE, BLEU）和人工评估

功能:
1. ROUGE 评估（ROUGE-1, ROUGE-2, ROUGE-L）
2. BLEU 评估
3. 分类准确率评估（多任务模式）
4. 生成人工评估模板
5. 统计分析与可视化
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import argparse
from datetime import datetime

# 评估指标
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed. Run: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk not installed. Run: pip install nltk")


class TicketEvaluator:
    """工单回复评估器"""

    def __init__(self, task_type: str = "multi_task"):
        self.task_type = task_type

        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

        if BLEU_AVAILABLE:
            self.smoothing = SmoothingFunction().method1

    def extract_classification(self, text: str) -> Dict[str, str]:
        """从多任务输出中提取分类结果"""
        result = {
            'type': None,
            'queue': None,
            'priority': None
        }

        # 匹配 "- Type: xxx" 格式
        type_match = re.search(r'Type:\s*(\w+)', text, re.IGNORECASE)
        if type_match:
            result['type'] = type_match.group(1)

        queue_match = re.search(r'Queue:\s*([^\n]+)', text, re.IGNORECASE)
        if queue_match:
            result['queue'] = queue_match.group(1).strip()

        priority_match = re.search(r'Priority:\s*(\w+)', text, re.IGNORECASE)
        if priority_match:
            result['priority'] = priority_match.group(1).lower()

        return result

    def extract_response(self, text: str) -> str:
        """从多任务输出中提取回复内容"""
        # 查找 "Response:" 后的内容
        match = re.search(r'Response:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """计算 ROUGE 分数"""
        if not ROUGE_AVAILABLE:
            return {}

        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }

    def compute_bleu(self, prediction: str, reference: str) -> float:
        """计算 BLEU 分数"""
        if not BLEU_AVAILABLE:
            return 0.0

        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]

        try:
            score = sentence_bleu(ref_tokens, pred_tokens,
                                  smoothing_function=self.smoothing)
            return score
        except:
            return 0.0

    def evaluate_sample(
        self,
        prediction: str,
        reference: str,
        reference_classification: Optional[Dict] = None
    ) -> Dict:
        """评估单个样本"""
        result = {}

        # 如果是多任务模式，分别评估分类和生成
        if self.task_type == "multi_task":
            # 提取分类
            pred_class = self.extract_classification(prediction)
            result['pred_classification'] = pred_class

            if reference_classification:
                result['classification_correct'] = {
                    'type': pred_class['type'] == reference_classification.get('type'),
                    'queue': pred_class['queue'] == reference_classification.get('queue'),
                    'priority': pred_class['priority'] == reference_classification.get('priority'),
                }

            # 提取回复
            pred_response = self.extract_response(prediction)
            ref_response = self.extract_response(reference)
        else:
            pred_response = prediction
            ref_response = reference

        # 计算 ROUGE
        rouge_scores = self.compute_rouge(pred_response, ref_response)
        result.update(rouge_scores)

        # 计算 BLEU
        result['bleu'] = self.compute_bleu(pred_response, ref_response)

        # 长度统计
        result['pred_length'] = len(pred_response.split())
        result['ref_length'] = len(ref_response.split())

        return result

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        reference_classifications: Optional[List[Dict]] = None
    ) -> Dict:
        """批量评估"""
        all_results = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            ref_class = reference_classifications[i] if reference_classifications else None
            result = self.evaluate_sample(pred, ref, ref_class)
            all_results.append(result)

        # 聚合结果
        aggregated = {}

        # ROUGE 平均分
        rouge_keys = ['rouge1_f', 'rouge2_f', 'rougeL_f']
        for key in rouge_keys:
            values = [r.get(key, 0) for r in all_results if key in r]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)

        # BLEU 平均分
        bleu_values = [r.get('bleu', 0) for r in all_results]
        aggregated['avg_bleu'] = np.mean(bleu_values)

        # 分类准确率（多任务模式）
        if self.task_type == "multi_task" and reference_classifications:
            for field in ['type', 'queue', 'priority']:
                correct = sum(
                    1 for r in all_results
                    if r.get('classification_correct', {}).get(field, False)
                )
                aggregated[f'{field}_accuracy'] = correct / len(all_results)

        # 长度统计
        aggregated['avg_pred_length'] = np.mean([r['pred_length'] for r in all_results])
        aggregated['avg_ref_length'] = np.mean([r['ref_length'] for r in all_results])

        return {
            'aggregated': aggregated,
            'samples': all_results
        }


def create_human_eval_template(
    samples: List[Dict],
    output_path: Path,
    n_samples: int = 100
) -> None:
    """
    创建人工评估模板

    评估维度:
    1. 相关性 (1-5): 回复是否解决了客户问题
    2. 专业性 (1-5): 回复是否专业、准确
    3. 完整性 (1-5): 回复是否完整、不遗漏信息
    4. 语气 (1-5): 回复是否礼貌、有同理心
    5. 总体质量 (1-5): 整体评分
    """
    # 随机采样
    if len(samples) > n_samples:
        import random
        samples = random.sample(samples, n_samples)

    rows = []
    for i, sample in enumerate(samples):
        rows.append({
            'id': i + 1,
            'input': sample.get('input', '')[:500],  # 截断
            'reference': sample.get('reference', '')[:500],
            'prediction': sample.get('prediction', '')[:500],
            'relevance': '',      # 相关性
            'professionalism': '',  # 专业性
            'completeness': '',   # 完整性
            'tone': '',           # 语气
            'overall': '',        # 总体
            'comments': ''        # 备注
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Human evaluation template saved to {output_path}")
    print(f"Total samples: {len(rows)}")


def generate_report(results: Dict, output_path: Path) -> None:
    """生成评估报告"""
    report = []
    report.append("=" * 60)
    report.append("工单回复生成 - 评估报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")

    agg = results['aggregated']

    # ROUGE 分数
    report.append("## 自动评估指标")
    report.append("")
    report.append("### ROUGE 分数")
    report.append(f"  ROUGE-1 F1: {agg.get('avg_rouge1_f', 0):.4f} (±{agg.get('std_rouge1_f', 0):.4f})")
    report.append(f"  ROUGE-2 F1: {agg.get('avg_rouge2_f', 0):.4f} (±{agg.get('std_rouge2_f', 0):.4f})")
    report.append(f"  ROUGE-L F1: {agg.get('avg_rougeL_f', 0):.4f} (±{agg.get('std_rougeL_f', 0):.4f})")
    report.append("")

    report.append("### BLEU 分数")
    report.append(f"  BLEU: {agg.get('avg_bleu', 0):.4f}")
    report.append("")

    # 分类准确率
    if 'type_accuracy' in agg:
        report.append("### 分类准确率")
        report.append(f"  Type: {agg.get('type_accuracy', 0):.2%}")
        report.append(f"  Queue: {agg.get('queue_accuracy', 0):.2%}")
        report.append(f"  Priority: {agg.get('priority_accuracy', 0):.2%}")
        report.append("")

    # 长度统计
    report.append("### 长度统计")
    report.append(f"  平均预测长度: {agg.get('avg_pred_length', 0):.1f} 词")
    report.append(f"  平均参考长度: {agg.get('avg_ref_length', 0):.1f} 词")
    report.append("")

    # 质量分析
    report.append("## 质量分析")
    report.append("")

    samples = results['samples']
    rouge_scores = [s.get('rougeL_f', 0) for s in samples]

    report.append("### ROUGE-L 分布")
    report.append(f"  优秀 (>0.5): {sum(1 for s in rouge_scores if s > 0.5)} 条 ({sum(1 for s in rouge_scores if s > 0.5)/len(samples):.1%})")
    report.append(f"  良好 (0.3-0.5): {sum(1 for s in rouge_scores if 0.3 <= s <= 0.5)} 条 ({sum(1 for s in rouge_scores if 0.3 <= s <= 0.5)/len(samples):.1%})")
    report.append(f"  一般 (<0.3): {sum(1 for s in rouge_scores if s < 0.3)} 条 ({sum(1 for s in rouge_scores if s < 0.3)/len(samples):.1%})")

    report.append("")
    report.append("=" * 60)

    # 保存报告
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ticket response generation')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--references', type=str, required=True,
                        help='Path to references JSON file (test set)')
    parser.add_argument('--task_type', type=str,
                        choices=['response_generation', 'multi_task'],
                        default='multi_task')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for results')
    parser.add_argument('--human_eval_samples', type=int, default=100,
                        help='Number of samples for human evaluation')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(args.references, 'r', encoding='utf-8') as f:
        references_data = json.load(f)

    # 提取预测和参考
    predictions = [p['output'] for p in predictions_data]
    references = [r['output'] for r in references_data]

    # 如果是多任务，提取分类信息
    reference_classifications = None
    if args.task_type == "multi_task":
        evaluator = TicketEvaluator(task_type="multi_task")
        reference_classifications = [
            evaluator.extract_classification(r['output'])
            for r in references_data
        ]
    else:
        evaluator = TicketEvaluator(task_type="response_generation")

    # 评估
    print("Evaluating...")
    results = evaluator.evaluate_batch(
        predictions, references, reference_classifications
    )

    # 生成报告
    report_path = output_dir / "evaluation_report.txt"
    generate_report(results, report_path)

    # 保存详细结果
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # 只保存可序列化的部分
        json.dump({
            'aggregated': results['aggregated'],
            'sample_count': len(results['samples'])
        }, f, indent=2)
    print(f"Detailed results saved to {results_path}")

    # 创建人工评估模板
    human_eval_samples = [
        {
            'input': references_data[i]['input'],
            'reference': references_data[i]['output'],
            'prediction': predictions_data[i]['output']
        }
        for i in range(min(len(predictions_data), args.human_eval_samples))
    ]
    human_eval_path = output_dir / "human_eval_template.csv"
    create_human_eval_template(human_eval_samples, human_eval_path, args.human_eval_samples)


if __name__ == "__main__":
    main()
