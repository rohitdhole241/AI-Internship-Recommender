"""
Collaborative Filtering Model using Cosine Similarity
Alternative to scikit-surprise (Python 3.13 compatible)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringModel:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        
    def fit(self, user_item_matrix):
        """
        Train collaborative filtering model
        Args:
            user_item_matrix: DataFrame with students as rows, internships as columns
        """
        self.user_item_matrix = user_item_matrix
        
        # Calculate user-user similarity using cosine similarity
        self.user_similarity = cosine_similarity(user_item_matrix)
        
        print(f"✅ Collaborative model trained on {user_item_matrix.shape[0]} users and {user_item_matrix.shape[1]} items")
        
    def predict(self, student_id, internship_id):
        """
        Predict rating for student-internship pair
        Args:
            student_id: Student ID
            internship_id: Internship ID
        Returns:
            Predicted rating (1-5 scale)
        """
        if student_id not in self.user_item_matrix.index:
            return 3.0  # Default neutral rating
            
        if internship_id not in self.user_item_matrix.columns:
            return 3.0  # Default neutral rating
        
        # Get index of student
        user_idx = self.user_item_matrix.index.get_loc(student_id)
        
        # Get similar users
        similar_users = self.user_similarity[user_idx]
        
        # Get ratings from similar users for this internship
        internship_ratings = self.user_item_matrix[internship_id].values
        
        # Weighted average of ratings from similar users
        numerator = np.dot(similar_users, internship_ratings)
        denominator = np.sum(np.abs(similar_users)) + 1e-9  # Avoid division by zero
        
        predicted_rating = numerator / denominator
        
        # Clip to valid rating range
        return np.clip(predicted_rating, 1.0, 5.0)
    
    def get_top_recommendations(self, student_id, top_n=5):
        """
        Get top N internship recommendations for a student
        Args:
            student_id: Student ID
            top_n: Number of recommendations
        Returns:
            List of (internship_id, predicted_rating) tuples
        """
        if student_id not in self.user_item_matrix.index:
            return []
        
        # Get all internships student hasn't rated
        student_ratings = self.user_item_matrix.loc[student_id]
        unrated_internships = student_ratings[student_ratings == 0].index.tolist()
        
        # Predict ratings for unrated internships
        predictions = []
        for internship_id in unrated_internships:
            pred_rating = self.predict(student_id, internship_id)
            predictions.append((internship_id, pred_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]
