from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from dataclasses import dataclass

@dataclass
class ContractExample:
    text: str
    label: Optional[str] = None
    metadata: Optional[Dict] = None

class ContractDataset(Dataset):
    """Dataset class for contract clause data.
    
    Handles loading, preprocessing and validation of contract text data.
    Supports train/val/test splits and batching for training.
    
    Args:
        data_dir: Path to data directory containing contract files
        mode: One of train, val, test
        **kwargs: Additional arguments
    """

    def __init__(self, 
                 data_dir: Union[str, Path],
                 mode: str = "train",
                 **kwargs) -> None:
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.examples = []
        self.validate()
        self.load_data()
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.examples[idx]
        return {
            "text": example.text,
            "label": example.label if example.label else None,
            "metadata": example.metadata if example.metadata else None
        }
        
    def validate(self) -> bool:
        """Validate the data directory and configuration.

        Returns:
            bool: True if validation passes
        
        Raises:
            FileNotFoundError: If data directory doesnt exist
            ValueError: If mode is invalid
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data path {self.data_dir} does not exist")
        
        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Invalid mode {self.mode}, must be one of: train, val, test")
            
        return True
        
    def load_data(self) -> None:
        """Load contract files from data directory."""
        for file_path in self.data_dir.glob("*.txt"):
            try:
                text = self._read_file(file_path)
                text = self.preprocess(text)
                example = ContractExample(
                    text=text,
                    metadata={"file_path": str(file_path)}
                )
                self.examples.append(example)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
                
    def _read_file(self, file_path: Path) -> str:
        """Read and return contents of a text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    def preprocess(self, text: str) -> str:
        """Clean and normalize input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text string
        """
        # Basic cleaning
        text = text.strip()
        # Normalize whitespace
        text = " ".join(text.split())
        return text
        
    def get_batch(self, batch_size: int) -> Dict[str, any]:
        """Get a batch of examples for training.
        
        Args:
            batch_size: Number of examples per batch
            
        Returns:
            Dict containing batch data
        """
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        batch = [self[i] for i in indices]
        return {
            "texts": [ex["text"] for ex in batch],
            "labels": [ex["label"] for ex in batch] if batch[0]["label"] is not None else None,
            "metadata": [ex["metadata"] for ex in batch] if batch[0]["metadata"] is not None else None
        }
        
    def split_data(self, val_ratio: float = 0.1, test_ratio: float = 0.1) -> None:
        """Split data into train/val/test sets.
        
        Args:
            val_ratio: Ratio of validation set size to total
            test_ratio: Ratio of test set size to total
        """
        n = len(self)
        indices = list(range(n))
        random.shuffle(indices)
        
        test_size = int(test_ratio * n)
        val_size = int(val_ratio * n)
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        if self.mode == "train":
            self.examples = [self.examples[i] for i in train_indices]
        elif self.mode == "val":
            self.examples = [self.examples[i] for i in val_indices]
        else:
            self.examples = [self.examples[i] for i in test_indices]
