"""
Feedback Handler Component
Manages user feedback and updates the learning agent
"""

import pandas as pd
import config
from datetime import datetime

class FeedbackHandler:
    def __init__(self):
        """Initialize feedback handler"""
        self.feedback_df = pd.read_csv(config.FEEDBACK_FILE)
        
    def add_feedback(self, student_id, internship_id, rating, feedback_text, would_recommend='Yes'):
        """
        Add new feedback from a student
        Args:
            student_id: Student ID
            internship_id: Internship ID
            rating: Rating (1-5)
            feedback_text: Textual feedback
            would_recommend: Yes/No/Maybe
        """
        new_feedback = {
            'student_id': student_id,
            'internship_id': internship_id,
            'rating': rating,
            'feedback_text': feedback_text,
            'completion_date': datetime.now().strftime('%Y-%m-%d'),
            'would_recommend': would_recommend
        }
        
        # Append to dataframe
        self.feedback_df = pd.concat([
            self.feedback_df,
            pd.DataFrame([new_feedback])
        ], ignore_index=True)
        
        # Save to CSV
        self.feedback_df.to_csv(config.FEEDBACK_FILE, index=False)
        
        print(f"✅ Feedback added for {student_id} → {internship_id}")
        
        return True
    
    def get_student_feedback(self, student_id):
        """Get all feedback from a specific student"""
        return self.feedback_df[self.feedback_df['student_id'] == student_id]
    
    def get_internship_feedback(self, internship_id):
        """Get all feedback for a specific internship"""
        return self.feedback_df[self.feedback_df['internship_id'] == internship_id]
    
    def get_average_rating(self, internship_id):
        """Get average rating for an internship"""
        internship_feedback = self.get_internship_feedback(internship_id)
        
        if len(internship_feedback) == 0:
            return 0.0
        
        return internship_feedback['rating'].mean()
    
    def update_models(self, recommender):
        """
        Update recommendation models with new feedback
        This implements the Learning Agent concept
        Args:
            recommender: HybridRecommender instance
        """
        print("🔄 Updating models with new feedback...")
        
        # Reload feedback data
        feedback_df = pd.read_csv(config.FEEDBACK_FILE)
        
        # Recreate user-item matrix
        from utils.preprocessing import create_user_item_matrix, save_processed_data
        
        user_item_matrix = create_user_item_matrix(feedback_df)
        
        # Retrain collaborative model
        recommender.collaborative_model.fit(user_item_matrix)
        
        print("✅ Models updated successfully! System is now smarter.")
        
        return True
