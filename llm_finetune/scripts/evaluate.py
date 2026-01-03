#!/usr/bin/env python3
"""
Evaluation Script - Supports automatic evaluation (ROUGE, BLEU) and human evaluation

Features:
1. ROUGE evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
2. BLEU evaluation
3. Classification accuracy evaluation (multi-task mode)
4. Human evaluation template generation
5. Statistical analysis and visualization
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

# Evaluation metrics
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
    """Ticket response evaluator"""

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
        """Extract classification results from multi-task output"""
        result = {
            'type': None,
            'queue': None,
            'priority': None
        }

        # Match "- Type: xxx" format
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
        """Extract response content from multi-task output"""
        # Find content after "Response:"
        match = re.search(r'Response:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
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
        """Compute BLEU score"""
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
        """Evaluate a single sample"""
        result = {}

        # For multi-task mode, evaluate classification and generation separately
        if self.task_type == "multi_task":
            # Extract classification
            pred_class = self.extract_classification(prediction)
            result['pred_classification'] = pred_class

            if reference_classification:
                result['classification_correct'] = {
                    'type': pred_class['type'] == reference_classification.get('type'),
                    'queue': pred_class['queue'] == reference_classification.get('queue'),
                    'priority': pred_class['priority'] == reference_classification.get('priority'),
                }

            # Extract response
            pred_response = self.extract_response(prediction)
            ref_response = self.extract_response(reference)
        else:
            pred_response = prediction
            ref_response = reference

        # Compute ROUGE
        rouge_scores = self.compute_rouge(pred_response, ref_response)
        result.update(rouge_scores)

        # Compute BLEU
        result['bleu'] = self.compute_bleu(pred_response, ref_response)

        # Length statistics
        result['pred_length'] = len(pred_response.split())
        result['ref_length'] = len(ref_response.split())

        return result

    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        reference_classifications: Optional[List[Dict]] = None
    ) -> Dict:
        """Batch evaluation"""
        all_results = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            ref_class = reference_classifications[i] if reference_classifications else None
            result = self.evaluate_sample(pred, ref, ref_class)
            all_results.append(result)

        # Aggregate results
        aggregated = {}

        # ROUGE average scores
        rouge_keys = ['rouge1_f', 'rouge2_f', 'rougeL_f']
        for key in rouge_keys:
            values = [r.get(key, 0) for r in all_results if key in r]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)

        # BLEU average score
        bleu_values = [r.get('bleu', 0) for r in all_results]
        aggregated['avg_bleu'] = np.mean(bleu_values)

        # Classification accuracy (multi-task mode)
        if self.task_type == "multi_task" and reference_classifications:
            for field in ['type', 'queue', 'priority']:
                correct = sum(
                    1 for r in all_results
                    if r.get('classification_correct', {}).get(field, False)
                )
                aggregated[f'{field}_accuracy'] = correct / len(all_results)

        # Length statistics
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
    Create human evaluation template

    Evaluation dimensions:
    1. Relevance (1-5): Does the response address the customer's issue
    2. Professionalism (1-5): Is the response professional and accurate
    3. Completeness (1-5): Is the response complete without missing information
    4. Tone (1-5): Is the response polite and empathetic
    5. Overall quality (1-5): Overall rating
    """
    # Random sampling
    if len(samples) > n_samples:
        import random
        samples = random.sample(samples, n_samples)

    rows = []
    for i, sample in enumerate(samples):
        rows.append({
            'id': i + 1,
            'input': sample.get('input', '')[:500],  # Truncate
            'reference': sample.get('reference', '')[:500],
            'prediction': sample.get('prediction', '')[:500],
            'relevance': '',
            'professionalism': '',
            'completeness': '',
            'tone': '',
            'overall': '',
            'comments': ''
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Human evaluation template saved to {output_path}")
    print(f"Total samples: {len(rows)}")


def generate_report(results: Dict, output_path: Path) -> None:
    """Generate evaluation report"""
    report = []
    report.append("=" * 60)
    report.append("Ticket Response Generation - Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")

    agg = results['aggregated']

    # ROUGE scores
    report.append("## Automatic Evaluation Metrics")
    report.append("")
    report.append("### ROUGE Scores")
    report.append(f"  ROUGE-1 F1: {agg.get('avg_rouge1_f', 0):.4f} (+/-{agg.get('std_rouge1_f', 0):.4f})")
    report.append(f"  ROUGE-2 F1: {agg.get('avg_rouge2_f', 0):.4f} (+/-{agg.get('std_rouge2_f', 0):.4f})")
    report.append(f"  ROUGE-L F1: {agg.get('avg_rougeL_f', 0):.4f} (+/-{agg.get('std_rougeL_f', 0):.4f})")
    report.append("")

    report.append("### BLEU Score")
    report.append(f"  BLEU: {agg.get('avg_bleu', 0):.4f}")
    report.append("")

    # Classification accuracy
    if 'type_accuracy' in agg:
        report.append("### Classification Accuracy")
        report.append(f"  Type: {agg.get('type_accuracy', 0):.2%}")
        report.append(f"  Queue: {agg.get('queue_accuracy', 0):.2%}")
        report.append(f"  Priority: {agg.get('priority_accuracy', 0):.2%}")
        report.append("")

    # Length statistics
    report.append("### Length Statistics")
    report.append(f"  Avg prediction length: {agg.get('avg_pred_length', 0):.1f} words")
    report.append(f"  Avg reference length: {agg.get('avg_ref_length', 0):.1f} words")
    report.append("")

    # Quality analysis
    report.append("## Quality Analysis")
    report.append("")

    samples = results['samples']
    rouge_scores = [s.get('rougeL_f', 0) for s in samples]

    report.append("### ROUGE-L Distribution")
    report.append(f"  Excellent (>0.5): {sum(1 for s in rouge_scores if s > 0.5)} samples ({sum(1 for s in rouge_scores if s > 0.5)/len(samples):.1%})")
    report.append(f"  Good (0.3-0.5): {sum(1 for s in rouge_scores if 0.3 <= s <= 0.5)} samples ({sum(1 for s in rouge_scores if 0.3 <= s <= 0.5)/len(samples):.1%})")
    report.append(f"  Fair (<0.3): {sum(1 for s in rouge_scores if s < 0.3)} samples ({sum(1 for s in rouge_scores if s < 0.3)/len(samples):.1%})")

    report.append("")
    report.append("=" * 60)

    # Save report
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

    # Load data
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(args.references, 'r', encoding='utf-8') as f:
        references_data = json.load(f)

    # Extract predictions and references
    predictions = [p['output'] for p in predictions_data]
    references = [r['output'] for r in references_data]

    # For multi-task, extract classification info
    reference_classifications = None
    if args.task_type == "multi_task":
        evaluator = TicketEvaluator(task_type="multi_task")
        reference_classifications = [
            evaluator.extract_classification(r['output'])
            for r in references_data
        ]
    else:
        evaluator = TicketEvaluator(task_type="response_generation")

    # Evaluate
    print("Evaluating...")
    results = evaluator.evaluate_batch(
        predictions, references, reference_classifications
    )

    # Generate report
    report_path = output_dir / "evaluation_report.txt"
    generate_report(results, report_path)

    # Save detailed results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # Only save serializable parts
        json.dump({
            'aggregated': results['aggregated'],
            'sample_count': len(results['samples'])
        }, f, indent=2)
    print(f"Detailed results saved to {results_path}")

    # Create human evaluation template
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
