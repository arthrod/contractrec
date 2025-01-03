import torch
import torch.nn as nn
from typing import Dict, List

class ClauseClassifier(nn.Module):
    """Classifier for identifying contract clause types."""
    
    def __init__(self, input_size: int, num_classes: int):
        """Initialize classifier model.
        
        Args:
            input_size: Size of input features
            num_classes: Number of clause classes
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        return self.classifier(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Class predictions
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ClauseClassifier":
        """Load pretrained classifier.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded classifier
        """
        model_dict = torch.load(model_path)
        model = cls(
            input_size=model_dict["input_size"],
            num_classes=model_dict["num_classes"]
        )
        model.load_state_dict(model_dict["state_dict"])
        return model
        
    def save_pretrained(self, model_path: str) -> None:
        """Save model weights and config.
        
        Args:
            model_path: Path to save model
        """
        model_dict = {
            "input_size": self.classifier[0].in_features,
            "num_classes": self.classifier[-1].out_features,
            "state_dict": self.state_dict()
        }
        torch.save(model_dict, model_path)
