"""
Hybrid Recommendation System combining Content-Based and Collaborative Filtering
"""

import pandas as pd
import config

class HybridModel:
    def __init__(self, content_model, collaborative_model):
        """
        Initialize hybrid model
        Args:
            content_model: Trained ContentBasedModel instance
            collaborative_model: Trained CollaborativeFilteringModel instance
        """
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        
    def get_recommendations(self, student_id, student_profile, internships_df, top_n=5):
        """
        Get hybrid recommendations
        Args:
            student_id: Student ID
            student_profile: Student skill profile string
            internships_df: DataFrame with all internships
            top_n: Number of recommendations
        Returns:
            DataFrame with top N recommendations and hybrid scores
        """
        # Get content-based recommendations
        content_recs = self.content_model.get_recommendations(student_profile, top_n=20)
        
        # Add collaborative scores
        collaborative_scores = []
        for internship_id in content_recs['internship_id']:
            collab_score = self.collaborative_model.predict(student_id, internship_id)
            # Normalize to 0-1 scale
            collaborative_scores.append(collab_score / 5.0)
        
        content_recs['collaborative_score'] = collaborative_scores
        
        # Calculate hybrid score using weighted combination
        content_recs['hybrid_score'] = (
            config.CONTENT_WEIGHT * content_recs['content_score'] +
            config.COLLABORATIVE_WEIGHT * content_recs['collaborative_score']
        )
        
        # Sort by hybrid score and return top N
        recommendations = content_recs.sort_values('hybrid_score', ascending=False).head(top_n)
        
        return recommendations
