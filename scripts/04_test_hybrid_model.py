"""
Test Hybrid Recommendation Model
"""

import sys
sys.path.append('.')

from components.recommender import HybridRecommender

def main():
    print("=" * 70)
    print("TESTING HYBRID RECOMMENDATION MODEL")
    print("=" * 70)
    
    # Initialize recommender
    recommender = HybridRecommender()
    
    # Test multiple students
    test_students = ['S001', 'S004', 'S007', 'S011']
    
    for student_id in test_students:
        print(f"\n{'=' * 70}")
        print(f"RECOMMENDATIONS FOR STUDENT: {student_id}")
        print(f"{'=' * 70}")
        
        recommendations = recommender.recommend(student_id, top_n=3)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['company']} - {rec['role']}")
            print(f"   Match Score: {rec['match_score']:.4f}")
            print(f"   Rating: {rec['rating']}⭐")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid model testing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
