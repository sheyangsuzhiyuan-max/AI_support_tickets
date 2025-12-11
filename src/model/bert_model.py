"""
BERT model implementation for text classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional


def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """
    Get tokenizer for the specified model.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        Tokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_name)


class BertClassifier(nn.Module):
    """
    BERT-based classifier for text classification.
    
    Uses a pretrained transformer model with a classification head.
    """
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 freeze_bert: bool = False):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Name of the pretrained model
            num_classes: Number of output classes
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT weights
        """
        super(BertClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from config
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        logits = self.classifier(pooled_output)
        
        return logits

