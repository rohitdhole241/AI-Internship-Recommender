"""
Utils Package
Helper functions for data processing and utilities
"""

from .data_loader import load_students_data, load_internships_data, load_feedback_data, load_all_data
from .preprocessing import (
    preprocess_students,
    preprocess_internships,
    preprocess_feedback,
    create_user_item_matrix,
    save_processed_data
)
from .similarity import (
    calculate_cosine_similarity,
    get_top_n_similar_items,
    calculate_weighted_score,
    normalize_scores
)

__all__ = [
    'load_students_data',
    'load_internships_data',
    'load_feedback_data',
    'load_all_data',
    'preprocess_students',
    'preprocess_internships',
    'preprocess_feedback',
    'create_user_item_matrix',
    'save_processed_data',
    'calculate_cosine_similarity',
    'get_top_n_similar_items',
    'calculate_weighted_score',
    'normalize_scores'
]
