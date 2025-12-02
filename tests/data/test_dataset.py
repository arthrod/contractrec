import pytest
import json
import torch
import numpy as np
from pathlib import Path
from src.data.dataset import LedgarDataset

# Mock Model for testing embedding methods
class MockModel:
    def get_hidden_size(self):
        return 8

    def encode_text(self, texts: list[str]):
        return torch.randn(len(texts), self.get_hidden_size())

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample JSON contract files."""
    data_dir = tmp_path / "contracts"
    data_dir.mkdir()

    # Valid contract 1
    contract1 = {
        "id": "contract001",
        "text": "This is the full text of contract 1.",
        "clauses": [
            {"text": "This is clause A.", "label": "ClauseA"},
            {"text": "This is clause B.", "label": "ClauseB"}
        ]
    }
    with open(data_dir / "contract1.json", "w") as f:
        json.dump(contract1, f)

    # Valid contract 2
    contract2 = {
        "id": "contract002",
        "text": "This is the full text of contract 2.",
        "clauses": [
            {"text": "This is another clause B.", "label": "ClauseB"},
            {"text": "This is clause C.", "label": "ClauseC"}
        ]
    }
    with open(data_dir / "contract2.json", "w") as f:
        json.dump(contract2, f)

    # Invalid contract (missing 'clauses')
    invalid_contract = {"id": "contract003", "text": "Invalid text."}
    with open(data_dir / "invalid.json", "w") as f:
        json.dump(invalid_contract, f)

    # Non-json file
    (data_dir / "notes.txt").write_text("This is not a contract.")
    
    return data_dir

def test_init_and_load_contracts(sample_data_dir):
    """Test dataset initialization and contract loading."""
    dataset = LedgarDataset(sample_data_dir)
    assert dataset.data_dir == sample_data_dir
    assert len(dataset.contracts) == 2  # Only valid contracts should be loaded
    assert dataset.contracts[0]['id'] == 'contract001'
    assert dataset.contracts[1]['id'] == 'contract002'

def test_validate_data():
    """Test the data validation logic."""
    dataset = LedgarDataset(Path(".")) # dummy path
    valid_contract = {"id": "1", "text": "...", "clauses": [{"text": "c1", "label": "l1"}]}
    invalid_contract_missing_key = {"id": "1", "text": "..."}
    invalid_contract_wrong_type = {"id": "1", "text": "...", "clauses": "not a list"}
    
    assert dataset.validate_data(valid_contract) is True
    assert dataset.validate_data(invalid_contract_missing_key) is False
    assert dataset.validate_data(invalid_contract_wrong_type) is False
    assert dataset.validate_data("not a dict") is False

def test_get_all_clause_types(sample_data_dir):
    """Test scanning for all unique clause types."""
    dataset = LedgarDataset(sample_data_dir)
    expected_types = sorted(["ClauseA", "ClauseB", "ClauseC"])
    assert dataset.all_clause_types == expected_types

def test_create_proxy_datasets(sample_data_dir):
    """Test the creation of proxy datasets."""
    # The default fixture data is too small for stratified splitting with the given ratios.
    # We need at least 4 relevant examples per clause type to ensure splits are possible.
    # Existing from fixture: ClauseA: 1, ClauseB: 2, ClauseC: 1.
    
    # We add more contracts to meet the minimum size.
    new_contracts = [
        {"id": "c03", "text": "...", "clauses": [{"label": "ClauseA", "text": "..."}]},
        {"id": "c04", "text": "...", "clauses": [{"label": "ClauseA", "text": "..."}]},
        {"id": "c05", "text": "...", "clauses": [{"label": "ClauseA", "text": "..."}]}, # A count is now 4
        {"id": "c06", "text": "...", "clauses": [{"label": "ClauseB", "text": "..."}]},
        {"id": "c07", "text": "...", "clauses": [{"label": "ClauseB", "text": "..."}]}, # B count is now 4
        {"id": "c08", "text": "...", "clauses": [{"label": "ClauseC", "text": "..."}]},
        {"id": "c09", "text": "...", "clauses": [{"label": "ClauseC", "text": "..."}]},
        {"id": "c10", "text": "...", "clauses": [{"label": "ClauseC", "text": "..."}]}, # C count is now 4
    ]

    for contract in new_contracts:
        with open(sample_data_dir / f"{contract['id']}.json", "w") as f:
            json.dump(contract, f)

    dataset = LedgarDataset(sample_data_dir)
    train, val, test = dataset.create_proxy_datasets(dataset.all_clause_types, train_ratio=0.6, val_ratio=0.2)
    
    assert isinstance(train, list)
    assert isinstance(val, list)
    assert isinstance(test, list)
    
    # Check that datasets are populated. If splitting fails for a clause type,
    # the code errors out, so if we get here, it worked.
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

def test_get_clause_embeddings(mock_model):
    """Test clause embedding generation."""
    dataset = LedgarDataset(Path("."))
    clauses = ["clause 1", "clause 2", "clause 3"]
    embeddings = dataset.get_clause_embeddings(mock_model, clauses)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, mock_model.get_hidden_size())

def test_get_contract_representation(mock_model):
    """Test contract representation generation."""
    dataset = LedgarDataset(Path("."))
    contract_text = "First sentence.\nSecond sentence."
    representation = dataset.get_contract_representation(contract_text, mock_model)
    assert isinstance(representation, torch.Tensor)
    assert representation.shape == (mock_model.get_hidden_size(),)
    
    # Test empty contract
    empty_rep = dataset.get_contract_representation("", mock_model)
    assert torch.equal(empty_rep, torch.zeros(mock_model.get_hidden_size()))

def test_get_clause_type_representation(sample_data_dir, mock_model):
    """Test clause type representation generation."""
    dataset = LedgarDataset(sample_data_dir)

    # Test existing clause type
    rep = dataset.get_clause_type_representation("ClauseB", mock_model)
    assert isinstance(rep, torch.Tensor)
    assert rep.shape == (mock_model.get_hidden_size(),)
    
    # Test non-existent clause type
    rep_none = dataset.get_clause_type_representation("NonExistentClause", mock_model)
    assert rep_none is None

def test_create_contract_clause_matrix(sample_data_dir):
    """Test the creation of the contract-clause matrix."""
    dataset = LedgarDataset(sample_data_dir)
    matrix = dataset.create_contract_clause_matrix()
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(dataset.contracts), len(dataset.all_clause_types))

    # Manually check matrix content based on sample_data_dir
    # Contracts are loaded sorted by name: contract1.json, contract2.json
    # Clause types are sorted: ClauseA, ClauseB, ClauseC
    # contract1 has A, B
    # contract2 has B, C
    expected_matrix = np.array([
        [1, 1, 0],
        [0, 1, 1]
    ])
    np.testing.assert_array_equal(matrix, expected_matrix)
