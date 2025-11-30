"""
Test Collaborative Filtering Model
"""

import sys
sys.path.append('.')

import pandas as pd
from models.collaborative import CollaborativeFilteringModel

def main():
    print("=" * 70)
    print("TESTING COLLABORATIVE FILTERING MODEL")
    print("=" * 70)
    
    # Load user-item matrix
    print("\n📂 Loading user-item matrix...")
    user_item_matrix = pd.read_csv('data/processed/user_item_matrix.csv', index_col=0)
    
    # Train model
    print("\n🤖 Training collaborative filtering model...")
    model = CollaborativeFilteringModel()
    model.fit(user_item_matrix)
    
    # Test predictions
    print("\n📊 Testing rating predictions for Student S001")
    print("-" * 70)
    
    test_student = 'S001'
    test_internships = ['I002', 'I014', 'I016']
    
    for internship_id in test_internships:
        predicted_rating = model.predict(test_student, internship_id)
        print(f"Internship {internship_id}: Predicted Rating = {predicted_rating:.2f}/5.0")
    
    # Get top recommendations
    print("\n🎯 Top 5 Collaborative Recommendations for S001:")
    print("-" * 70)
    
    recommendations = model.get_top_recommendations(test_student, top_n=5)
    
    for i, (internship_id, rating) in enumerate(recommendations, 1):
        print(f"{i}. {internship_id}: Predicted Rating = {rating:.2f}/5.0")
    
    print("\n" + "=" * 70)
    print("✅ Collaborative filtering model testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
