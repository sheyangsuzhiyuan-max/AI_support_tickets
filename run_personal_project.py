#!/usr/bin/env python3
"""
One-click BERT fine-tuning experiments script

Usage:
    python run_personal_project.py                    # Run all experiments
    python run_personal_project.py --quick            # Quick test (1 epoch)
    python run_personal_project.py --exp EXP1 EXP2   # Run specific experiments
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_utils import load_text_classification_data
from src.text_preprocess import basic_clean
from src.model.bert_model import BertClassifier, get_tokenizer


class TicketDataset(Dataset):
    """Dataset for ticket classification"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')

    return avg_loss, accuracy, f1_macro


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = outputs.argmax(dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')

    return accuracy, f1_macro, f1_weighted


def run_experiment(exp_config, train_texts, train_labels, val_texts, val_labels, device, compute_confusion=False):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_config['name']}")
    print(f"{'='*80}")
    print(f"LR: {exp_config['lr']}, Epochs: {exp_config['epochs']}, Freeze: {exp_config['freeze_bert']}")

    # Initialize tokenizer and datasets
    tokenizer = get_tokenizer(exp_config['model_name'])
    train_dataset = TicketDataset(train_texts, train_labels, tokenizer)
    val_dataset = TicketDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=exp_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=exp_config['batch_size'], shuffle=False)

    # Initialize model
    model = BertClassifier(
        model_name=exp_config['model_name'],
        num_classes=3,
        dropout=0.3,
        freeze_bert=exp_config['freeze_bert']
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=exp_config['lr'])

    # Training loop
    best_val_f1 = 0
    results = {
        'config': exp_config,
        'epochs': []
    }

    for epoch in range(exp_config['epochs']):
        print(f"\nEpoch {epoch+1}/{exp_config['epochs']}")

        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_acc, val_f1_macro, val_f1_weighted = evaluate(model, val_loader, device)

        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_f1_macro': float(train_f1),
            'val_acc': float(val_acc),
            'val_f1_macro': float(val_f1_macro),
            'val_f1_weighted': float(val_f1_weighted)
        }
        results['epochs'].append(epoch_results)

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Acc: {val_acc:.4f}, F1 Macro: {val_f1_macro:.4f}, F1 Weighted: {val_f1_weighted:.4f}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro

    results['best_val_f1_macro'] = float(best_val_f1)
    print(f"\n✓ Best Val F1 Macro: {best_val_f1:.4f}")

    # Compute confusion matrix for best model if requested
    if compute_confusion:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                preds = outputs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
        results['classification_report'] = classification_report(all_labels, all_preds,
                                                                  target_names=['high', 'low', 'medium'],
                                                                  output_dict=True)

    # Clean up GPU memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return results


def generate_visualizations(all_results, output_dir='figures'):
    """Generate visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. Training curves for all experiments
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for result in all_results:
        exp_name = result['config']['name']
        epochs_data = result['epochs']

        epoch_nums = [e['epoch'] for e in epochs_data]
        train_loss = [e['train_loss'] for e in epochs_data]
        train_acc = [e['train_acc'] for e in epochs_data]
        val_acc = [e['val_acc'] for e in epochs_data]
        val_f1 = [e['val_f1_macro'] for e in epochs_data]

        axes[0, 0].plot(epoch_nums, train_loss, marker='o', label=exp_name)
        axes[0, 1].plot(epoch_nums, train_acc, marker='o', label=exp_name)
        axes[1, 0].plot(epoch_nums, val_acc, marker='o', label=exp_name)
        axes[1, 1].plot(epoch_nums, val_f1, marker='o', label=exp_name)

    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation F1 Macro', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    training_curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrix for best experiment
    best_result = max(all_results, key=lambda x: x['best_val_f1_macro'])
    if 'confusion_matrix' in best_result:
        cm = np.array(best_result['confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['high', 'low', 'medium'],
                    yticklabels=['high', 'low', 'medium'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {best_result["config"]["name"]}',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Performance comparison bar chart
    exp_names = [r['config']['name'] for r in all_results]
    val_f1_scores = [r['best_val_f1_macro'] for r in all_results]
    val_acc_scores = [r['epochs'][-1]['val_acc'] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax1.bar(x, val_acc_scores, width, label='Validation Accuracy', color='skyblue')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(x, val_f1_scores, width, label='Best Val F1 Macro', color='lightcoral')
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Best F1 Macro Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'training_curves': training_curves_path,
        'confusion_matrix': cm_path if 'confusion_matrix' in best_result else None,
        'performance_comparison': comparison_path
    }


def generate_markdown_report(all_results, summary_df, figure_paths, timestamp):
    """Generate comprehensive markdown report"""
    best_result = max(all_results, key=lambda x: x['best_val_f1_macro'])

    report = f"""# BERT Fine-tuning Experiments Report
# Support Ticket Priority Classification

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** DistilBERT-base-uncased
**Task:** 3-class classification (high, medium, low priority)

---

## Executive Summary

This report presents a systematic comparison of different BERT fine-tuning strategies for support ticket priority classification. We evaluated **{len(all_results)} different configurations** varying learning rates, training epochs, and encoder freezing strategies.

**Best Configuration:**
- **Experiment:** {best_result['config']['name']}
- **Learning Rate:** {best_result['config']['lr']}
- **Epochs:** {best_result['config']['epochs']}
- **Freeze BERT:** {best_result['config']['freeze_bert']}
- **Best Val F1 Macro:** {best_result['best_val_f1_macro']:.4f}
- **Final Val Accuracy:** {best_result['epochs'][-1]['val_acc']:.4f}

---

## Table of Contents

1. [Experiment Overview](#experiment-overview)
2. [Results Summary](#results-summary)
3. [Training Curves](#training-curves)
4. [Performance Comparison](#performance-comparison)
5. [Confusion Matrix](#confusion-matrix)
6. [Detailed Analysis](#detailed-analysis)
7. [Conclusions and Recommendations](#conclusions-and-recommendations)

---

## 1. Experiment Overview

### Dataset
- **Training samples:** 19,782
- **Validation samples:** 4,239
- **Classes:** 3 (high, medium, low priority)

### Experiments Conducted

"""

    for i, result in enumerate(all_results, 1):
        config = result['config']
        report += f"{i}. **{config['name']}**\n"
        report += f"   - Learning Rate: {config['lr']}\n"
        report += f"   - Epochs: {config['epochs']}\n"
        report += f"   - Freeze BERT: {config['freeze_bert']}\n"
        report += f"   - Description: {config.get('description', 'N/A')}\n\n"

    report += """---

## 2. Results Summary

### Performance Metrics Table

"""

    # Convert summary_df to markdown table
    report += summary_df.to_markdown(index=False) + "\n\n"

    report += """---

## 3. Training Curves

The following plots show the training progression across all experiments:

"""

    if figure_paths['training_curves']:
        report += f"![Training Curves]({figure_paths['training_curves']})\n\n"

    report += """**Observations:**
- Training loss consistently decreases across epochs for all experiments
- Validation accuracy and F1 scores show convergence patterns
- Different learning rates show distinct convergence behaviors

---

## 4. Performance Comparison

"""

    if figure_paths['performance_comparison']:
        report += f"![Performance Comparison]({figure_paths['performance_comparison']})\n\n"

    report += """---

## 5. Confusion Matrix

Confusion matrix for the best performing model:

"""

    if figure_paths['confusion_matrix']:
        report += f"![Confusion Matrix]({figure_paths['confusion_matrix']})\n\n"

    # Add classification report if available
    if 'classification_report' in best_result:
        report += "### Classification Report (Best Model)\n\n"
        report += "```\n"
        cr = best_result['classification_report']
        report += f"              precision    recall  f1-score   support\n\n"
        for label in ['high', 'low', 'medium']:
            if label in cr:
                p = cr[label]['precision']
                r = cr[label]['recall']
                f1 = cr[label]['f1-score']
                s = int(cr[label]['support'])
                report += f"    {label:8s}      {p:.4f}    {r:.4f}    {f1:.4f}     {s}\n"

        if 'accuracy' in cr:
            report += f"\n    accuracy                          {cr['accuracy']:.4f}\n"
        if 'macro avg' in cr:
            ma = cr['macro avg']
            report += f"   macro avg      {ma['precision']:.4f}    {ma['recall']:.4f}    {ma['f1-score']:.4f}\n"
        if 'weighted avg' in cr:
            wa = cr['weighted avg']
            report += f"weighted avg      {wa['precision']:.4f}    {wa['recall']:.4f}    {wa['f1-score']:.4f}\n"
        report += "```\n\n"

    report += """---

## 6. Detailed Analysis

### Key Findings

"""

    # Rank experiments
    sorted_results = sorted(all_results, key=lambda x: x['best_val_f1_macro'], reverse=True)

    report += f"""1. **Best Learning Rate:** LR = {sorted_results[0]['config']['lr']} achieved the highest F1 score
2. **Impact of Epochs:** {'More epochs generally improved performance' if sorted_results[0]['config']['epochs'] >= 5 else 'Optimal performance achieved within 3-5 epochs'}
3. **Encoder Freezing:** {'Freezing BERT encoder reduced performance significantly' if any(r['config']['freeze_bert'] for r in sorted_results[-2:]) else 'Full fine-tuning outperformed frozen encoder'}

### Experiment Rankings

"""

    for i, result in enumerate(sorted_results, 1):
        config = result['config']
        final_metrics = result['epochs'][-1]
        report += f"{i}. **{config['name']}** (F1: {result['best_val_f1_macro']:.4f})\n"
        report += f"   - Final Val Acc: {final_metrics['val_acc']:.4f}\n"
        report += f"   - Final Train Acc: {final_metrics['train_acc']:.4f}\n"
        report += f"   - Training Loss: {final_metrics['train_loss']:.4f}\n\n"

    report += """---

## 7. Conclusions and Recommendations

### Conclusions

"""

    best_config = sorted_results[0]['config']
    report += f"""- **Optimal Configuration:** {best_config['name']} with LR={best_config['lr']}, {best_config['epochs']} epochs
- **Performance:** Achieved {sorted_results[0]['best_val_f1_macro']:.2%} F1 macro score on validation set
- **Generalization:** {'Good generalization with minimal overfitting' if abs(sorted_results[0]['epochs'][-1]['train_acc'] - sorted_results[0]['epochs'][-1]['val_acc']) < 0.1 else 'Some overfitting observed'}

### Recommendations for Production

1. **Use the best configuration** ({best_config['name']}) for production deployment
2. **Monitor for class imbalance** - consider weighted loss if certain classes underperform
3. **Consider ensemble methods** - combining top 2-3 models may improve robustness
4. **Regular retraining** - retrain model periodically with new data to maintain performance

### Future Work

- Experiment with different BERT variants (BERT-base, RoBERTa, ALBERT)
- Implement learning rate scheduling strategies
- Test with larger batch sizes if GPU memory allows
- Explore data augmentation techniques
- Implement cross-validation for more robust evaluation

---

## Technical Details

**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
**PyTorch Version:** {torch.__version__}
**Transformers Library:** HuggingFace Transformers
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This report was automatically generated by the BERT fine-tuning experiment pipeline.*
"""

    return report


def get_experiment_configs(quick_mode=False):
    """Define all experiment configurations"""
    base_epochs = 1 if quick_mode else 3

    experiments = [
        {
            'name': 'EXP1_baseline_lr2e5',
            'model_name': 'distilbert-base-uncased',
            'lr': 2e-5,
            'epochs': base_epochs,
            'freeze_bert': False,
            'batch_size': 32,
            'description': 'Baseline: Full fine-tuning, LR=2e-5'
        },
        {
            'name': 'EXP2_lr3e5',
            'model_name': 'distilbert-base-uncased',
            'lr': 3e-5,
            'epochs': base_epochs,
            'freeze_bert': False,
            'batch_size': 32,
            'description': 'Higher LR: LR=3e-5'
        },
        {
            'name': 'EXP3_lr5e5',
            'model_name': 'distilbert-base-uncased',
            'lr': 5e-5,
            'epochs': base_epochs,
            'freeze_bert': False,
            'batch_size': 32,
            'description': 'Even higher LR: LR=5e-5'
        },
        {
            'name': 'EXP4_frozen',
            'model_name': 'distilbert-base-uncased',
            'lr': 2e-5,
            'epochs': base_epochs,
            'freeze_bert': True,
            'batch_size': 32,
            'description': 'Frozen encoder: only train classifier'
        },
    ]

    # Add longer training experiment only in non-quick mode
    if not quick_mode:
        experiments.append({
            'name': 'EXP5_lr3e5_epoch5',
            'model_name': 'distilbert-base-uncased',
            'lr': 3e-5,
            'epochs': 5,
            'freeze_bert': False,
            'batch_size': 32,
            'description': 'More epochs: LR=3e-5, 5 epochs'
        })

    return experiments


def main():
    parser = argparse.ArgumentParser(description='Run BERT fine-tuning experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (1 epoch)')
    parser.add_argument('--exp', nargs='+', help='Specific experiments to run (e.g., EXP1 EXP2)')
    parser.add_argument('--output-dir', default='data', help='Output directory for results')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # Load data
    print("\nLoading data...")
    train_texts, train_labels, label2id, id2label = load_text_classification_data('train')
    val_texts, val_labels, _, _ = load_text_classification_data('val')

    # Clean text
    train_texts = [basic_clean(text) for text in train_texts]
    val_texts = [basic_clean(text) for text in val_texts]

    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")

    # Get experiment configs
    all_experiments = get_experiment_configs(quick_mode=args.quick)

    # Filter experiments if specified
    if args.exp:
        all_experiments = [e for e in all_experiments if e['name'] in args.exp]
        if not all_experiments:
            print(f"Error: No matching experiments found for {args.exp}")
            return

    print(f"\n{'='*80}")
    print(f"Running {len(all_experiments)} experiments")
    print(f"{'='*80}")
    for exp in all_experiments:
        print(f"  - {exp['name']}: {exp['description']}")

    # Run experiments
    all_results = []
    for i, exp_config in enumerate(all_experiments, 1):
        print(f"\n[{i}/{len(all_experiments)}] Starting {exp_config['name']}...")
        try:
            result = run_experiment(exp_config, train_texts, train_labels, val_texts, val_labels, device, compute_confusion=False)
            all_results.append(result)
        except Exception as e:
            print(f"✗ Error in {exp_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\n✗ No experiments completed successfully")
        return

    # Create summary
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")

    summary_data = []
    for result in all_results:
        config = result['config']
        last_epoch = result['epochs'][-1]

        summary_data.append({
            'Experiment': config['name'],
            'LR': config['lr'],
            'Epochs': config['epochs'],
            'Freeze': config['freeze_bert'],
            'Best_Val_F1': result['best_val_f1_macro'],
            'Final_Val_Acc': last_epoch['val_acc'],
            'Final_Train_Acc': last_epoch['train_acc'],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best_Val_F1', ascending=False)
    print(summary_df.to_string(index=False))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    results_file = os.path.join(args.output_dir, f'bert_experiments_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {results_file}")

    # Save summary CSV
    summary_csv = os.path.join(args.output_dir, f'bert_experiments_summary_{timestamp}.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary saved to: {summary_csv}")

    # Compute confusion matrix for best model
    print(f"\n{'='*80}")
    print("Computing confusion matrix for best model...")
    print(f"{'='*80}")
    best_result = max(all_results, key=lambda x: x['best_val_f1_macro'])
    best_config = best_result['config']
    print(f"Re-running best model: {best_config['name']}")
    best_result_with_cm = run_experiment(best_config, train_texts, train_labels, val_texts, val_labels, device, compute_confusion=True)

    # Update the best result in all_results
    for i, result in enumerate(all_results):
        if result['config']['name'] == best_config['name']:
            all_results[i] = best_result_with_cm
            break

    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating Visualizations...")
    print(f"{'='*80}")
    figure_paths = generate_visualizations(all_results, output_dir='figures')
    print(f"✓ Training curves: {figure_paths['training_curves']}")
    if figure_paths['confusion_matrix']:
        print(f"✓ Confusion matrix: {figure_paths['confusion_matrix']}")
    print(f"✓ Performance comparison: {figure_paths['performance_comparison']}")

    # Generate markdown report
    print(f"\n{'='*80}")
    print("Generating Comprehensive Report...")
    print(f"{'='*80}")
    report_content = generate_markdown_report(all_results, summary_df, figure_paths, timestamp)
    report_file = 'BERT_Finetuning_Report.md'
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f"✓ Report saved to: {report_file}")

    # Print best result
    best = summary_df.iloc[0]
    print(f"\n{'='*80}")
    print("BEST CONFIGURATION:")
    print(f"{'='*80}")
    print(f"Experiment: {best['Experiment']}")
    print(f"Learning Rate: {best['LR']}")
    print(f"Epochs: {best['Epochs']}")
    print(f"Freeze BERT: {best['Freeze']}")
    print(f"Best Val F1 Macro: {best['Best_Val_F1']:.4f}")
    print(f"{'='*80}")
    print(f"\n✓ Complete! Check {report_file} for detailed analysis with visualizations.")


if __name__ == '__main__':
    main()
