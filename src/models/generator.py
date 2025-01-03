from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ClauseGenerator(nn.Module):
    """Generator model for creating contract clauses."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize generator model.
        
        Args:
            model_name: Name of pretrained model to use
        """
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through generator.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for training
            
        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values
        }
    
    def generate_clause(
        self,
        prompt: str,
        max_length: int = 256,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> List[str]:
        """Generate clause text from prompt.
        
        Args:
            prompt: Text prompt to condition generation
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated clause texts
        """
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Generate text
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ClauseGenerator":
        """Load pretrained generator.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded generator model
        """
        model = cls()
        model.load_state_dict(torch.load(model_path))
        return model
        
    def save_pretrained(self, model_path: str) -> None:
        """Save model weights.
        
        Args:
            model_path: Path to save model
        """
        torch.save(self.state_dict(), model_path)
