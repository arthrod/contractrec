from typing import List, Dict, Optional
import numpy as np
import faiss
import torch
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilter:
    """Collaborative filtering recommender for contract clauses."""
    
    def __init__(self, n_factors: int = 100):
        """Initialize collaborative filter.
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings: np.ndarray) -> None:
        """Train collaborative filter on ratings matrix.
        
        Args:
            ratings: User-item ratings matrix
        """
        # SVD decomposition
        U, s, Vh = np.linalg.svd(
            ratings,
            full_matrices=False
        )
        
        # Get latent factors
        s_root = np.sqrt(s[:self.n_factors])
        self.user_factors = U[:, :self.n_factors] * s_root
        self.item_factors = Vh[:self.n_factors, :].T * s_root[:, np.newaxis]
        
    def predict(self, user_idx: int, item_idx: Optional[int] = None) -> np.ndarray:
        """Predict ratings for user-item pairs.
        
        Args:
            user_idx: User index
            item_idx: Optional item index (if None, predict all items)
            
        Returns:
            Predicted ratings
        """
        if item_idx is None:
            return self.user_factors[user_idx] @ self.item_factors.T
        return self.user_factors[user_idx] @ self.item_factors[item_idx]
        
    def recommend(self, user_idx: int, n_items: int = 10) -> List[int]:
        """Get top-N recommendations for user.
        
        Args:
            user_idx: User index
            n_items: Number of items to recommend
            
        Returns:
            List of recommended item indices
        """
        # Predict ratings for all items
        pred_ratings = self.predict(user_idx)
        
        # Get top-N items
        return np.argsort(pred_ratings)[-n_items:]
        
    def similar_items(self, item_idx: int, n_items: int = 10) -> List[int]:
        """Find similar items using latent factors.
        
        Args:
            item_idx: Reference item index
            n_items: Number of similar items to return
            
        Returns:
            List of similar item indices
        """
        # Compute similarities between items
        sims = cosine_similarity(
            self.item_factors[item_idx].reshape(1, -1),
            self.item_factors
        )
        
        # Get top-N similar items
        return np.argsort(sims[0])[-n_items:]
        
    def save(self, filepath: str) -> None:
        """Save model factors.
        
        Args:
            filepath: Path to save model
        """
        np.savez(
            filepath,
            user_factors=self.user_factors,
            item_factors=self.item_factors
        )
        
    @classmethod
    def load(cls, filepath: str) -> "CollaborativeFilter":
        """Load saved model.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model
        """
        data = np.load(filepath)
        model = cls(n_factors=data["user_factors"].shape[1])
        model.user_factors = data["user_factors"]
        model.item_factors = data["item_factors"]
        return model
