import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from transformers import BertModel, BertTokenizer

class ContractBERT(nn.Module):
    """BERT-based model for contract understanding with additional tasks.
    
    Implements the base BERT model with additional tasks:
    1. Clause type classification
    2. Predicting if words in clause label belong to clause
    3. Predicting if two sentences belong to same clause
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2):
        """Initialize ContractBERT model.
        
        Args:
            model_name: Name of pretrained BERT model to use
            num_labels: Number of clause types to predict
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        
        # Classification heads
        hidden_size = self.bert.config.hidden_size
        self.clause_type_classifier = nn.Linear(hidden_size, num_labels)
        self.label_word_predictor = nn.Linear(hidden_size, 2)  # Binary prediction
        self.sentence_pair_classifier = nn.Linear(hidden_size, 2)  # Binary prediction
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_word_labels: Optional[torch.Tensor] = None,
        sentence_pair_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            labels: Optional clause type labels
            label_word_labels: Optional labels for word prediction task
            sentence_pair_labels: Optional labels for sentence pair task
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output      # [batch_size, hidden_size]
        hidden_states = outputs.hidden_states
        
        # Dropout on pooled output
        pooled_output = self.dropout(pooled_output)
        
        # Clause type classification
        clause_type_logits = self.clause_type_classifier(pooled_output)
        
        # Label word prediction (using sequence output)
        label_word_logits = self.label_word_predictor(sequence_output)
        
        # Sentence pair classification
        sentence_pair_logits = self.sentence_pair_classifier(pooled_output)
        
        # Calculate losses if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            clause_type_loss = loss_fct(clause_type_logits.view(-1, self.num_labels), labels.view(-1))
            loss = clause_type_loss
            
        if label_word_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            label_word_loss = loss_fct(label_word_logits.view(-1, 2), label_word_labels.view(-1))
            loss = loss + label_word_loss if loss is not None else label_word_loss
            
        if sentence_pair_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sentence_pair_loss = loss_fct(sentence_pair_logits.view(-1, 2), sentence_pair_labels.view(-1))
            loss = loss + sentence_pair_loss if loss is not None else sentence_pair_loss
            
        return {
            "loss": loss,
            "clause_type_logits": clause_type_logits,
            "label_word_logits": label_word_logits,
            "sentence_pair_logits": sentence_pair_logits,
            "hidden_states": hidden_states
        }
        
    def encode_text(self, text: str, tokenizer: BertTokenizer) -> Dict[str, torch.Tensor]:
        """Encode text input using tokenizer.
        
        Args:
            text: Input text to encode
            tokenizer: BERT tokenizer
            
        Returns:
            Dictionary of encoded inputs
        """
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return encoded
        
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "ContractBERT":
        """Load model from pretrained weights.
        
        Args:
            model_path: Path to pretrained model
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Loaded ContractBERT model
        """
        model = cls(**kwargs)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model
        
    def save_pretrained(self, model_path: str) -> None:
        """Save model weights.
        
        Args:
            model_path: Path to save model
        """
        torch.save(self.state_dict(), model_path)
