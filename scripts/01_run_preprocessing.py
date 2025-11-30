"""
Script to load, explore, and preprocess all datasets
Run this first before building models
"""

import pandas as pd
import sys
sys.path.append('.')

from utils.data_loader import load_all_data
from utils.preprocessing import (
    preprocess_students, 
    preprocess_internships, 
    preprocess_feedback, 
    create_user_item_matrix,
    save_processed_data
)

def main():
    print("=" * 70)
    print("AI-BASED HYBRID RECOMMENDATION SYSTEM - DATA PREPROCESSING")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n📂 STEP 1: LOADING DATA")
    print("-" * 70)
    students, internships, feedback = load_all_data()
    
    # Step 2: Explore Data
    print("\n📊 STEP 2: DATA EXPLORATION")
    print("-" * 70)
    
    print("\n📌 STUDENTS DATA:")
    print(f"  - Total students: {len(students)}")
    print(f"  - Branches: {students['branch'].unique()}")
    print(f"  - Average CGPA: {students['cgpa'].mean():.2f}")
    print(f"  - Sample student skills: {students['skills'].iloc[0]}")
    
    print("\n📌 INTERNSHIPS DATA:")
    print(f"  - Total internships: {len(internships)}")
    print(f"  - Companies: {internships['company'].nunique()}")
    print(f"  - Domains: {internships['domain'].unique()}")
    print(f"  - Average rating: {internships['rating'].mean():.2f}")
    print(f"  - Stipend range: ₹{internships['stipend'].min()} - ₹{internships['stipend'].max()}")
    
    print("\n📌 FEEDBACK DATA:")
    print(f"  - Total feedback entries: {len(feedback)}")
    print(f"  - Rating distribution:")
    for rating, count in feedback['rating'].value_counts().sort_index().items():
        print(f"    {rating} stars: {count} reviews")
    print(f"  - Recommendation rate: {(feedback['would_recommend'] == 'Yes').mean() * 100:.1f}%")
    
    # Step 3: Preprocess Data
    print("\n🔧 STEP 3: PREPROCESSING DATA")
    print("-" * 70)
    students_processed = preprocess_students(students)
    internships_processed = preprocess_internships(internships)
    feedback_processed = preprocess_feedback(feedback)
    user_item_matrix = create_user_item_matrix(feedback_processed)
    
    # Step 4: Save Processed Data
    print("\n💾 STEP 4: SAVING PROCESSED DATA")
    print("-" * 70)
    save_processed_data(students_processed, internships_processed, feedback_processed, user_item_matrix)
    
    print("\n" + "=" * 70)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python scripts/02_test_content_model.py")
    print("  2. Run: python scripts/03_test_collaborative_model.py")
    print("  3. Run: python scripts/04_test_hybrid_model.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
