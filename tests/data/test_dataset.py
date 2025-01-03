"""
Tests for the LedgarDataset class
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.dataset import LedgarDataset

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing"""
    df = pd.DataFrame({
        'text': ['This is a test contract.', 'Another contract text here.'],
        'label': ['test_label', 'another_label']
    })
    file_path = tmp_path / "test_contracts.csv"
    df.to_csv(file_path, index=False)
    return tmp_path, file_path

def test_init(tmp_path):
    """Test dataset initialization"""
    dataset = LedgarDataset(tmp_path)
    assert dataset.data_path == tmp_path
    assert dataset.max_length == 512
    
    with pytest.raises(ValueError):
        LedgarDataset("nonexistent/path")

def test_load_contracts(sample_data):
    """Test contract loading"""
    tmp_path, file_path = sample_data
    dataset = LedgarDataset(tmp_path)
    
    contracts = dataset.load_contracts(file_path.name)
    assert len(contracts) == 2
    assert all(col in contracts.columns for col in ['text', 'label'])
    
    with pytest.raises(FileNotFoundError):
        dataset.load_contracts("nonexistent.csv")

def test_preprocess_text():
    """Test text preprocessing"""
    dataset = LedgarDataset(Path("."))
    
    text = " This is a TEST contract... with SOME noise!!! "
    processed = dataset.preprocess_text(text)
    
    assert processed == "this is a test contract with some noise"
    assert dataset.preprocess_text(None) == ""
    assert dataset.preprocess_text(123) == ""

def test_extract_clauses():
    """Test clause extraction"""
    dataset = LedgarDataset(Path("."))
    
    text = "First clause. Second clause; Third clause."
    clauses = dataset.extract_clauses(text)
    
    assert len(clauses) == 3
    assert all(isinstance(c, str) for c in clauses)

def test_tokenize_text():
    """Test text tokenization"""
    dataset = LedgarDataset(Path("."))
    
    result = dataset.tokenize_text("Test contract text")
    assert "input_ids" in result
    assert "attention_mask" in result
    
    with pytest.raises(ValueError):
        dataset.tokenize_text("")
    with pytest.raises(ValueError):
        dataset.tokenize_text(None)

def test_validate_data(sample_data):
    """Test data validation"""
    tmp_path, file_path = sample_data
    dataset = LedgarDataset(tmp_path)
    
    # Should raise error before loading data
    with pytest.raises(ValueError):
        dataset.validate_data()
        
    dataset.load_contracts(file_path.name)
    assert dataset.validate_data() is True
