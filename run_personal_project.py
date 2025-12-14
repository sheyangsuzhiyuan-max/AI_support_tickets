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
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

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


def run_experiment(exp_config, train_texts, train_labels, val_texts, val_labels, device):
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

    # Clean up GPU memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return results


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
            result = run_experiment(exp_config, train_texts, train_labels, val_texts, val_labels, device)
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


if __name__ == '__main__':
    main()
