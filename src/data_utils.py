"""
Data loading utilities for support ticket classification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os


def load_split(split: str) -> pd.DataFrame:
    """
    Load a data split CSV file.
    
    Args:
        split: One of 'train', 'val', 'test'
        
    Returns:
        DataFrame with the loaded data
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'processed', f'tickets_{split}.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def load_text_classification_data(split: str) -> Tuple[List[str], np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Load text classification data for a given split.
    
    Args:
        split: One of 'train', 'val', 'test'
        
    Returns:
        Tuple of:
        - texts: List of text strings
        - labels: NumPy array of label IDs (0, 1, 2)
        - label2id: Dictionary mapping label names to IDs
        - id2label: Dictionary mapping IDs to label names
    """
    df = load_split(split)
    
    # Extract text column
    if 'text' not in df.columns:
        raise ValueError("Column 'text' not found in dataframe")
    texts = df['text'].fillna('').astype(str).tolist()
    
    # Extract priority labels
    if 'priority' not in df.columns:
        raise ValueError("Column 'priority' not found in dataframe")
    priority_labels = df['priority'].fillna('medium').astype(str).tolist()
    
    # Create label mappings
    unique_labels = sorted(set(priority_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert labels to IDs
    labels = np.array([label2id[label] for label in priority_labels], dtype=np.int64)
    
    return texts, labels, label2id, id2label

