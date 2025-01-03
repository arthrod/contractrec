from typing import List, Dict, Optional
import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer

class SimilarityRecommender:
    """Document similarity-based recommender for contract clauses."""
    
    def __init__(self, model_name: str = "bert-base-uncased", index_type: str = "l2"):
        """Initialize similarity recommender.
        
        Args:
            model_name: Name of pretrained model for embeddings
            index_type: Type of FAISS index (l2 or cosine)
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.index = None
        self.index_type = index_type
        self.documents = []
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get document embeddings using pretrained model.
        
        Args:
            texts: List of document texts
            
        Returns:
            Document embedding matrix
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get embeddings
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0].detach().numpy()
        
        # Normalize if using cosine similarity
        if self.index_type == "cosine":
            faiss.normalize_L2(embeddings)
            
        return embeddings
        
    def fit(self, documents: List[str]) -> None:
        """Build search index from documents.
        
        Args:
            documents: List of document texts
        """
        self.documents = documents
        embeddings = self._get_embeddings(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        if self.index_type == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            
        self.index.add(embeddings)
        
    def recommend(self, query: str, k: int = 10) -> List[Dict]:
        """Get similar documents for query.
        
        Args:
            query: Query text
            k: Number of recommendations
            
        Returns:
            List of recommendations with scores
        """
        # Get query embedding
        query_emb = self._get_embeddings([query])
        
        # Search index
        scores, indices = self.index.search(query_emb, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "document": self.documents[idx],
                "score": float(score)
            })
            
        return results
        
    def save(self, index_path: str) -> None:
        """Save FAISS index.
        
        Args:
            index_path: Path to save index
        """
        faiss.write_index(self.index, index_path)
        
    @classmethod
    def load(cls, index_path: str, documents: List[str], model_name: str = "bert-base-uncased") -> "SimilarityRecommender":
        """Load saved recommender.
        
        Args:
            index_path: Path to saved index
            documents: List of documents
            model_name: Name of pretrained model
            
        Returns:
            Loaded recommender
        """
        recommender = cls(model_name=model_name)
        recommender.documents = documents
        recommender.index = faiss.read_index(index_path)
        return recommender
