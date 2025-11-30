"""
Unit tests for recommendation models
"""

import sys
sys.path.append('.')

import pandas as pd
from models.content_based import ContentBasedModel
from models.collaborative import CollaborativeFilteringModel
from models.hybrid import HybridModel
import config

def test_content_based_model():
    """Test content-based recommendation model"""
    print("\n🧪 Testing Content-Based Model...")
    
    # Load data
    internships = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv")
    
    # Initialize and train
    model = ContentBasedModel()
    model.fit(internships)
    
    # Test recommendation
    test_profile = "Python Machine Learning Data Science SQL"
    recommendations = model.get_recommendations(test_profile, top_n=3)
    
    assert len(recommendations) == 3, "Should return 3 recommendations"
    assert 'content_score' in recommendations.columns, "Should have content_score column"
    assert recommendations['content_score'].iloc[0] >= recommendations['content_score'].iloc[1], "Should be sorted by score"
    
    print("✅ Content-based model test passed!")
    return True

def test_collaborative_model():
    """Test collaborative filtering model"""
    print("\n🧪 Testing Collaborative Filtering Model...")
    
    # Load data
    user_item_matrix = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/user_item_matrix.csv", index_col=0)
    
    # Initialize and train
    model = CollaborativeFilteringModel()
    model.fit(user_item_matrix)
    
    # Test prediction
    prediction = model.predict('S001', 'I002')
    
    assert 1.0 <= prediction <= 5.0, "Prediction should be in valid rating range"
    
    # Test recommendations
    recommendations = model.get_top_recommendations('S001', top_n=5)
    
    assert len(recommendations) <= 5, "Should return at most 5 recommendations"
    
    print("✅ Collaborative filtering model test passed!")
    return True

def test_hybrid_model():
    """Test hybrid recommendation model"""
    print("\n🧪 Testing Hybrid Model...")
    
    # Load data
    internships = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv")
    user_item_matrix = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/user_item_matrix.csv", index_col=0)
    
    # Initialize models
    content_model = ContentBasedModel()
    content_model.fit(internships)
    
    collab_model = CollaborativeFilteringModel()
    collab_model.fit(user_item_matrix)
    
    hybrid_model = HybridModel(content_model, collab_model)
    
    # Test recommendation
    test_profile = "Python Machine Learning SQL"
    recommendations = hybrid_model.get_recommendations('S001', test_profile, internships, top_n=5)
    
    assert len(recommendations) == 5, "Should return 5 recommendations"
    assert 'hybrid_score' in recommendations.columns, "Should have hybrid_score column"
    
    print("✅ Hybrid model test passed!")
    return True

def run_all_tests():
    """Run all unit tests"""
    print("=" * 70)
    print("RUNNING UNIT TESTS FOR RECOMMENDATION SYSTEM")
    print("=" * 70)
    
    try:
        test_content_based_model()
        test_collaborative_model()
        test_hybrid_model()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    run_all_tests()
