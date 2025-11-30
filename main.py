"""
Main entry point for the AI-Based Hybrid Recommendation System
"""

import sys
sys.path.append('.')

from utils.data_loader import load_all_data
from components.recommender import HybridRecommender
import config

def get_recommendations_for_student(student_id):
    """Get recommendations for a specific student"""
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDATIONS FOR STUDENT: {student_id}")
    print(f"{'=' * 70}")
    
    # Initialize recommender
    recommender = HybridRecommender()
    
    # Get recommendations
    recommendations = recommender.recommend(
        student_id=student_id,
        top_n=config.TOP_N_RECOMMENDATIONS
    )
    
    print(f"\n🎯 Top {config.TOP_N_RECOMMENDATIONS} Recommended Internships:")
    print("-" * 70)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['company']} - {rec['role']}")
        print(f"   Domain: {rec['domain']}")
        print(f"   Location: {rec['location']}")
        print(f"   Duration: {rec['duration_months']} months")
        print(f"   Stipend: ₹{rec['stipend']}")
        print(f"   Rating: {rec['rating']}⭐ ({rec['total_reviews']} reviews)")
        print(f"   Match Score: {rec['match_score']:.2f}")
    
    return recommendations

def main():
    print("=" * 70)
    print("AI-BASED HYBRID RECOMMENDATION SYSTEM FOR INTERNSHIPS")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    students, internships, feedback = load_all_data()
    
    # Example: Get recommendations for student S001 (Rohit)
    recommendations = get_recommendations_for_student('S001')
    
    print(f"\n{'=' * 70}")
    print("✅ Recommendation generation complete!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
