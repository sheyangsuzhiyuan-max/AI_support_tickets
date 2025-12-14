"""
Evaluation utilities for text classification.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
from typing import Dict, Any


def evaluate_classification(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           average: str = "macro") -> Dict[str, Any]:
    """
    Evaluate classification performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for F1 score ('macro', 'micro', 'weighted')

    Returns:
        Dictionary containing:
        - accuracy: Accuracy score
        - f1_macro: Macro-averaged F1 score
        - f1_micro: Micro-averaged F1 score
        - f1_weighted: Weighted F1 score
        - precision_macro: Macro-averaged precision
        - recall_macro: Macro-averaged recall
        - report: Classification report string
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    report = classification_report(y_true, y_pred)

    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'report': report
    }

    return results

