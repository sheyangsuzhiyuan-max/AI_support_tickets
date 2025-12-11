"""
TextCNN model implementation for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN model for text classification.
    
    Architecture:
    - Embedding layer (with optional pretrained embeddings)
    - Multiple 1D convolutional filters with different kernel sizes
    - Max-over-time pooling
    - Fully connected layers with dropout
    - Output layer for classification
    """
    
    def __init__(self, vocab_size: int, 
                 embed_dim: int = 128,
                 num_filters: int = 100,
                 filter_sizes: list = [3, 4, 5],
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 padding_idx: int = 0,
                 pretrained_embeddings: torch.Tensor = None):
        """
        Initialize TextCNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes (kernel sizes)
            num_classes: Number of output classes
            dropout: Dropout probability
            padding_idx: Index for padding tokens
            pretrained_embeddings: Optional pretrained embedding matrix (vocab_size, embed_dim)
        """
        super(TextCNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # Freeze padding token embedding
            self.embedding.weight.data[padding_idx].fill_(0)
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                     out_channels=num_filters, 
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        # Total features = num_filters * len(filter_sizes)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # x shape: (batch_size, seq_len)
        # Embedding: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        
        # Transpose for Conv1d: (batch_size, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv1d: (batch_size, num_filters, conv_seq_len)
            conv_out = F.relu(conv(x))
            # Max pooling: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        # (batch_size, num_filters * len(filter_sizes))
        x = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer
        # (batch_size, num_classes)
        logits = self.fc(x)
        
        return logits
