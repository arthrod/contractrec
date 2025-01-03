from typing import Dict, List, Union, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None
) -> Dict[str, Any]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names
        
    Returns:
        Dictionary of metrics
    """
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted"
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    # Per-class metrics if labels provided
    if labels is not None:
        per_class = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None
        )
        for i, label in enumerate(labels):
            metrics[f"{label}_precision"] = float(per_class[0][i])
            metrics[f"{label}_recall"] = float(per_class[1][i])
            metrics[f"{label}_f1"] = float(per_class[2][i])
            metrics[f"{label}_support"] = int(per_class[3][i])
            
    return metrics

def compute_similarity_metrics(
    similarities: List[float],
    relevance: List[int],
    k: int = None
) -> Dict[str, float]:
    """Compute similarity search metrics.
    
    Args:
        similarities: List of similarity scores
        relevance: Binary relevance labels
        k: Optional cutoff for top-k metrics
        
    Returns:
        Dictionary of metrics
    """
    if k is None:
        k = len(similarities)
    
    # Sort by similarity
    sorted_idx = np.argsort(similarities)[::-1][:k]
    relevance_at_k = [relevance[i] for i in sorted_idx]
    
    # Compute metrics
    precision = np.mean(relevance_at_k)
    dcg = np.sum([rel/np.log2(i+2) for i, rel in enumerate(relevance_at_k)])
    idcg = np.sum([1/np.log2(i+2) for i in range(sum(relevance_at_k))])
    ndcg = dcg/idcg if idcg > 0 else 0
    
    return {
        f"precision@{k}": float(precision),
        f"ndcg@{k}": float(ndcg)
    }

def compute_generation_metrics(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """Compute text generation metrics.
    
    Args:
        references: Reference texts
        hypotheses: Generated texts
        
    Returns:
        Dictionary of metrics
    """
    from rouge_score import rouge_scorer
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    
    # Compute ROUGE scores
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores.append({
            "rouge1_f": score["rouge1"].fmeasure,
            "rouge2_f": score["rouge2"].fmeasure,
            "rougeL_f": score["rougeL"].fmeasure
        })
    
    # Average scores
    avg_scores = {}
    for key in scores[0].keys():
        avg_scores[key] = float(np.mean([s[key] for s in scores]))
        
    return avg_scores
