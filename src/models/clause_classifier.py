import torch
import torch.nn as nn
from typing import Dict, Optional

class ClauseClassifier(nn.Module):
    """Neural classifier for contract clause types.
    
    Implements a multi-layer neural network for classifying contract clauses
    into predefined types.
    """
    
    def __init__(self, input_size: int, num_classes: int):
        """Initialize clause classifier.
        
        Args:
            input_size: Size of input features
            num_classes: Number of clause types to predict
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Logits tensor of shape [batch_size, num_classes]
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
        
    def training_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Execute one training step.
        
        Args:
            batch: Dictionary containing input and labels
            optimizer: Optimizer instance
            
        Returns:
            Dictionary with training metrics
        """
        self.train()
        optimizer.zero_grad()
        
        x = batch["input"]
        y = batch["label"]
        
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}
        
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one validation step.
        
        Args:
            batch: Dictionary containing input and labels
            
        Returns:
            Dictionary with validation metrics
        """
        self.eval()
        with torch.no_grad():
            x = batch["input"]
            y = batch["label"]
            
            logits = self.forward(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            
            return {"val_loss": loss.item(), "val_acc": acc.item()}
        
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "ClauseClassifier":
        """Load model from pretrained weights.
        
        Args:
            model_path: Path to pretrained model
            **kwargs: Additional arguments passed to __init__
            
        Returns:
            Loaded ClauseClassifier model
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
