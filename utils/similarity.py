"""
Utility functions for similarity calculations
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors
    Args:
        vector1: First vector (can be numpy array or list)
        vector2: Second vector (can be numpy array or list)
    Returns:
        Similarity score between 0 and 1
    """
    return cosine_similarity(
        np.array(vector1).reshape(1, -1),
        np.array(vector2).reshape(1, -1)
    )[0][0]

def get_top_n_similar_items(similarity_matrix, item_index, top_n=5):
    """
    Get top N most similar items from similarity matrix
    Args:
        similarity_matrix: 2D similarity matrix
        item_index: Index of the target item
        top_n: Number of similar items to return
    Returns:
        List of (index, similarity_score) tuples
    """
    # Get similarity scores for the item
    similarities = similarity_matrix[item_index]
    
    # Get indices sorted by similarity (excluding the item itself)
    similar_indices = np.argsort(similarities)[::-1]
    similar_indices = [idx for idx in similar_indices if idx != item_index]
    
    # Return top N with scores
    top_similar = []
    for idx in similar_indices[:top_n]:
        top_similar.append((idx, similarities[idx]))
    
    return top_similar

def calculate_weighted_score(scores, weights):
    """
    Calculate weighted average of multiple scores
    Args:
        scores: List of scores
        weights: List of weights (must sum to 1.0)
    Returns:
        Weighted average score
    """
    if len(scores) != len(weights):
        raise ValueError("Scores and weights must have same length")
    
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError("Weights must sum to 1.0")
    
    return sum(s * w for s, w in zip(scores, weights))

def normalize_scores(scores, min_val=0, max_val=1):
    """
    Normalize scores to a given range
    Args:
        scores: List or array of scores
        min_val: Minimum value of range
        max_val: Maximum value of range
    Returns:
        Normalized scores
    """
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.full_like(scores, (min_val + max_val) / 2)
    
    normalized = (scores - min_score) / (max_score - min_score)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized
