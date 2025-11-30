"""
Content-Based Filtering using TF-IDF and Cosine Similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.internship_vectors = None
        self.internships_df = None
        
    def fit(self, internships_df):
        """
        Train content-based model
        Args:
            internships_df: DataFrame with internship_profile column
        """
        self.internships_df = internships_df
        
        # Create TF-IDF vectors from internship profiles
        self.internship_vectors = self.vectorizer.fit_transform(
            internships_df['internship_profile']
        )
        
        print(f"✅ Content-based model trained on {len(internships_df)} internships")
        
    def get_recommendations(self, student_profile, top_n=5):
        """
        Get top N recommendations based on student profile
        Args:
            student_profile: String containing student skills and interests
            top_n: Number of recommendations
        Returns:
            DataFrame with top N internships and similarity scores
        """
        # Transform student profile to vector
        student_vector = self.vectorizer.transform([student_profile])
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(student_vector, self.internship_vectors)[0]
        
        # Get top N indices
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        
        # Return internships with scores
        recommendations = self.internships_df.iloc[top_indices].copy()
        recommendations['content_score'] = similarity_scores[top_indices]
        
        return recommendations
