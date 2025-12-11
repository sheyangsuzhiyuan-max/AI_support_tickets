"""
Feature extraction utilities for text classification.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional


def build_tfidf_vectorizer(max_features: int = 50000, 
                          ngram_range: tuple = (1, 2), 
                          min_df: int = 5) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer for text feature extraction.
    
    Args:
        max_features: Maximum number of features to keep
        ngram_range: Range of n-grams to extract (min, max)
        min_df: Minimum document frequency for a term to be included
        
    Returns:
        Configured TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        stop_words='english',
        token_pattern=r'(?u)\b\w+\b'
    )
    
    return vectorizer

