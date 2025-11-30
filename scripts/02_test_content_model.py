"""
Test Content-Based Recommendation Model
"""

import sys
sys.path.append('.')

import pandas as pd
from models.content_based import ContentBasedModel

def main():
    print("=" * 70)
    print("TESTING CONTENT-BASED RECOMMENDATION MODEL")
    print("=" * 70)
    
    # Load processed data
    internships = pd.read_csv('data/processed/internships_processed.csv')
    students = pd.read_csv('data/processed/students_processed.csv')
    
    # Train model
    print("\n🤖 Training content-based model...")
    model = ContentBasedModel()
    model.fit(internships)
    
    # Test with student S001 (Rohit - Data Science interest)
    print("\n📊 Testing recommendations for Student S001 (Rohit)")
    print("-" * 70)
    
    student = students[students['student_id'] == 'S001']
    student_profile = student['skill_profile'].values[0]
    
    print(f"Student Profile: {student_profile[:100]}...")
    
    recommendations = model.get_recommendations(student_profile, top_n=5)
    
    print("\n🎯 Top 5 Content-Based Recommendations:")
    print("-" * 70)
    
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{i}. {rec['company']} - {rec['role']}")
        print(f"   Domain: {rec['domain']}")
        print(f"   Required Skills: {rec['required_skills'][:60]}...")
        print(f"   Content Match Score: {rec['content_score']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Content-based model testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
