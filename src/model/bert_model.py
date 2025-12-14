import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional


from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "distilbert-base-uncased"):
    """
    Get tokenizer for the specified model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier (default: distilbert-base-uncased)
    """
    return AutoTokenizer.from_pretrained(model_name)



class BertClassifier(nn.Module):
    """
    BERT-based classifier for text classification.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super(BertClassifier, self).__init__()

        self.model_name = model_name

        # Load BERT from HuggingFace Hub
        self.bert = AutoModel.from_pretrained(model_name)

        # Load config from HuggingFace Hub
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
