import pytest
import torch
from src.data.dataset import LedgarDataset
from src.models.contract_bert import ContractBERT
from src.models.classifier import ClauseClassifier
from src.models.generator import ClauseGenerator
from src.recommenders.collaborative import CollaborativeFilter
from src.recommenders.similarity import SimilarityRecommender
from src.utils.metrics import compute_classification_metrics
from src.utils.preprocessing import clean_text, extract_clauses

@pytest.fixture
def sample_text():
    return """Section 1. Test Clause
This is a test clause for testing purposes.

Section 2. Another Clause
This is another test clause."""

@pytest.fixture
def dataset(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return LedgarDataset(str(data_dir))

def test_dataset_initialization(dataset):
    assert dataset is not None
    assert isinstance(dataset.tokenizer, object)

def test_contract_bert():
    model = ContractBERT()
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "bert")
    assert hasattr(model, "classifier")

def test_clause_classifier():
    classifier = ClauseClassifier(768, 2)
    assert isinstance(classifier, torch.nn.Module)
    
    x = torch.randn(2, 768)
    output = classifier(x)
    assert output.shape == (2, 2)

def test_clause_generator():
    generator = ClauseGenerator()
    assert isinstance(generator, torch.nn.Module)
    assert hasattr(generator, "model")
    assert hasattr(generator, "tokenizer")

def test_collaborative_filter():
    cf = CollaborativeFilter(n_factors=10)
    assert cf.n_factors == 10
    assert cf.user_factors is None
    assert cf.item_factors is None

def test_similarity_recommender():
    recommender = SimilarityRecommender()
    assert isinstance(recommender.model, torch.nn.Module)
    assert recommender.index is None

def test_clean_text(sample_text):
    cleaned = clean_text(sample_text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0

def test_extract_clauses(sample_text):
    clauses = extract_clauses(sample_text)
    assert len(clauses) == 2
    assert all(isinstance(c, dict) for c in clauses)
    assert all("text" in c for c in clauses)
