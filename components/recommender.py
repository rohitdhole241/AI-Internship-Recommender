"""
Main Hybrid Recommender Component
Integrates all models and provides recommendation interface
"""

import pandas as pd
import config
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeFilteringModel
from models.hybrid import HybridModel

class HybridRecommender:
    def __init__(self):
        """Initialize the hybrid recommender system"""
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_model = None
        self.students_df = None
        self.internships_df = None
        
        # Load processed data
        self._load_data()
        
        # Train models
        self._train_models()
        
    def _load_data(self):
        """Load preprocessed data"""
        print("📂 Loading processed data...")
        
        self.students_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/students_processed.csv")
        self.internships_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv")
        user_item_matrix = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/user_item_matrix.csv", index_col=0)
        
        print(f"✅ Loaded {len(self.students_df)} students, {len(self.internships_df)} internships")
        
        return user_item_matrix
        
    def _train_models(self):
        """Train all recommendation models"""
        print("\n🤖 Training recommendation models...")
        
        # Load user-item matrix
        user_item_matrix = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/user_item_matrix.csv", index_col=0)
        
        # Train content-based model
        self.content_model = ContentBasedModel()
        self.content_model.fit(self.internships_df)
        
        # Train collaborative filtering model
        self.collaborative_model = CollaborativeFilteringModel()
        self.collaborative_model.fit(user_item_matrix)
        
        # Create hybrid model
        self.hybrid_model = HybridModel(self.content_model, self.collaborative_model)
        
        print("✅ All models trained successfully!")
        
    def recommend(self, student_id, top_n=5):
        """
        Get top N internship recommendations for a student
        Args:
            student_id: Student ID (e.g., 'S001')
            top_n: Number of recommendations to return
        Returns:
            List of dictionaries containing internship details
        """
        # Get student profile
        student = self.students_df[self.students_df['student_id'] == student_id]
        
        if student.empty:
            print(f"❌ Student {student_id} not found!")
            return []
        
        student_profile = student['skill_profile'].values[0]
        
        # Get hybrid recommendations
        recommendations = self.hybrid_model.get_recommendations(
            student_id, 
            student_profile, 
            self.internships_df, 
            top_n
        )
        
        # Convert to list of dictionaries
        result = []
        for _, row in recommendations.iterrows():
            result.append({
                'internship_id': row['internship_id'],
                'company': row['company'],
                'role': row['role'],
                'domain': row['domain'],
                'location': row['location'],
                'duration_months': row['duration_months'],
                'stipend': row['stipend'],
                'rating': row['rating'],
                'total_reviews': row['total_reviews'],
                'match_score': row['hybrid_score']
            })
        
        return result
