"""
PyTorch training utilities for neural network models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


def train_epoch_with_scheduler(dataloader, model, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # IMPORTANT: step AFTER optimizer.step, once per batch

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def eval_epoch_bert(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc, np.array(preds_all), np.array(labels_all)


def train_epoch(dataloader: DataLoader, 
                model: nn.Module, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        dataloader: DataLoader for training data
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Move batch to device
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
        else:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def eval_epoch(dataloader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataset.
    
    Args:
        dataloader: DataLoader for evaluation data
        model: PyTorch model
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        
    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_labels)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
            else:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

