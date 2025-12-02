from pathlib import Path
import os
import json
import glob
import random
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass
import logging
from sklearn.model_selection import train_test_split

from src.utils.text_preprocessor import TextPreprocessor

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContractExample:
    text: str
    label: Optional[str] = None
    metadata: Optional[Dict] = None

class LedgarDataset(Dataset):
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
        self.preprocessor = TextPreprocessor()
        self.all_clause_types = self._get_all_clause_types()
        self.contracts = self.load_contracts()
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.examples[idx]
        return {
            "text": example.text,
            "label": example.label if example.label else None,
            "metadata": example.metadata if example.metadata else None
        }
        
    def _get_all_clause_types(self) -> List[str]:
        """Scans data_dir for clause labels."""
        clause_types = set()
        for file_path in self.data_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "clauses" in data:
                    for clause in data["clauses"]:
                        if "label" in clause:
                            clause_types.add(clause["label"])
        return sorted(list(clause_types))

    def validate_data(self, contract: Dict) -> bool:
        """Validate a single contract dictionary."""
        if not isinstance(contract, dict):
            return False
        if "id" not in contract or "text" not in contract or "clauses" not in contract:
            return False
        if not isinstance(contract["clauses"], list):
            return False
        for clause in contract["clauses"]:
            if "text" not in clause or "label" not in clause:
                return False
        return True

    def load_contracts(self, file_pattern: str = "*.json") -> List[Dict]:
        """Load contract files from data directory."""
        contracts = []
        for file_path in self.data_dir.glob(file_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contract = json.load(f)
                if self.validate_data(contract):
                    contracts.append(contract)
                else:
                    logger.warning(f"Skipping invalid contract file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        return contracts
        
    def create_proxy_datasets(self, clause_types: List[str], train_ratio=0.6, val_ratio=0.2) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create proxy datasets for training, validation, and testing."""
        train_data, val_data, test_data = [], [], []

        for clause_type in clause_types:
            relevant_examples = []
            non_relevant_examples = []

            for contract in self.contracts:
                clauses = contract.get("clauses", [])
                contract_clause_labels = [c["label"] for c in clauses]

                if clause_type in contract_clause_labels:
                    # Create a copy of the contract with the relevant clause removed
                    new_contract = contract.copy()
                    new_contract["clauses"] = [c for c in clauses if c["label"] != clause_type]
                    new_contract["label"] = clause_type
                    relevant_examples.append(new_contract)
                else:
                    non_relevant_examples.append(contract)

            # Balance the dataset
            num_relevant = len(relevant_examples)
            if len(non_relevant_examples) > num_relevant:
                non_relevant_examples = random.sample(non_relevant_examples, num_relevant)
            
            all_examples = relevant_examples + non_relevant_examples
            labels = [1] * len(relevant_examples) + [0] * len(non_relevant_examples)

            if not all_examples:
                continue

            # Split the data
            train_val_examples, test_examples, train_val_labels, _ = train_test_split(
                all_examples, labels, test_size=(1 - train_ratio - val_ratio), random_state=42, stratify=labels
            )
            train_examples, val_examples, _, _ = train_test_split(
                train_val_examples, train_val_labels, test_size=(val_ratio / (train_ratio + val_ratio)), random_state=42, stratify=train_val_labels
            )

            train_data.extend(train_examples)
            val_data.extend(val_examples)
            test_data.extend(test_examples)

        return train_data, val_data, test_data
        
    def get_clause_embeddings(self, model, clauses: List[str], batch_size: int = 32) -> torch.Tensor:
        """Get clause embeddings using a ContractBERT model."""
        all_embeddings = []
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i+batch_size]
            embeddings = model.encode_text(batch)
            all_embeddings.append(embeddings.detach().cpu())
        return torch.cat(all_embeddings)
        
    def get_contract_representation(self, contract_text: str, model) -> torch.Tensor:
        """Get the mean [CLS] embedding of a contract's clauses."""
        # Simple paragraph splitting as a proxy for clause extraction
        clauses = [p.strip() for p in contract_text.split('\n') if p.strip()]
        if not clauses:
            return torch.zeros(model.get_hidden_size())

        embeddings = self.get_clause_embeddings(model, clauses)
        return embeddings.mean(dim=0)
        
    def get_clause_type_representation(self, clause_type: str, model) -> torch.Tensor:
        """Get the mean embedding of a clause type."""
        clause_texts = []
        for contract in self.contracts:
            for clause in contract.get("clauses", []):
                if clause.get("label") == clause_type:
                    clause_texts.append(clause["text"])
        
        if not clause_texts:
            return None

        return self.get_clause_embeddings(model, clause_texts).mean(dim=0)
        
    def create_contract_clause_matrix(self) -> np.ndarray:
        """Create a contract x clause_type binary presence matrix."""
        num_contracts = len(self.contracts)
        num_clause_types = len(self.all_clause_types)
        clause_type_to_idx = {ct: i for i, ct in enumerate(self.all_clause_types)}
        
        matrix = np.zeros((num_contracts, num_clause_types), dtype=np.int8)
        
        for i, contract in enumerate(self.contracts):
            for clause in contract.get("clauses", []):
                clause_label = clause.get("label")
                if clause_label in clause_type_to_idx:
                    j = clause_type_to_idx[clause_label]
                    matrix[i, j] = 1

        return matrix
