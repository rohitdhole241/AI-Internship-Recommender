import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def preprocess_students(df):
    """Clean and preprocess student data"""
    # Create a copy to avoid modifying original
    students_clean = df.copy()
    
    # Handle missing past_internships (replace 'None' with empty string)
    students_clean['past_internships'] = students_clean['past_internships'].replace('None', '')
    
    # Normalize CGPA to 0-1 scale
    scaler = MinMaxScaler()
    students_clean['cgpa_normalized'] = scaler.fit_transform(students_clean[['cgpa']])
    
    # Create combined skill profile (skills + domain_interest)
    students_clean['skill_profile'] = (
        students_clean['skills'] + ' ' + students_clean['domain_interest']
    )
    
    print(f"✅ Preprocessed {len(students_clean)} student records")
    return students_clean

def preprocess_internships(df):
    """Clean and preprocess internship data"""
    # Create a copy
    internships_clean = df.copy()
    
    # Normalize rating to 0-1 scale
    scaler = MinMaxScaler()
    internships_clean['rating_normalized'] = scaler.fit_transform(internships_clean[['rating']])
    
    # Create combined internship profile (required_skills + domain + role)
    internships_clean['internship_profile'] = (
        internships_clean['required_skills'] + ' ' + 
        internships_clean['domain'] + ' ' + 
        internships_clean['role']
    )
    
    print(f"✅ Preprocessed {len(internships_clean)} internship records")
    return internships_clean

def preprocess_feedback(df):
    """Clean and preprocess feedback data"""
    # Create a copy
    feedback_clean = df.copy()
    
    # Convert would_recommend to binary (1 for Yes, 0 for No/Maybe)
    feedback_clean['recommend_binary'] = feedback_clean['would_recommend'].apply(
        lambda x: 1 if x == 'Yes' else 0
    )
    
    # Ensure ratings are within valid range
    feedback_clean = feedback_clean[
        (feedback_clean['rating'] >= 1) & (feedback_clean['rating'] <= 5)
    ]
    
    print(f"✅ Preprocessed {len(feedback_clean)} feedback records")
    return feedback_clean

def create_user_item_matrix(feedback_df):
    """Create user-item rating matrix for collaborative filtering"""
    # Pivot table: rows = students, columns = internships, values = ratings
    user_item_matrix = feedback_df.pivot_table(
        index='student_id',
        columns='internship_id',
        values='rating',
        fill_value=0  # Fill missing ratings with 0
    )
    
    print(f"✅ Created user-item matrix: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} items")
    return user_item_matrix

def save_processed_data(students_df, internships_df, feedback_df, user_item_matrix):
    """Save preprocessed data to processed folder"""
    import config
    
    students_df.to_csv(f"{config.PROCESSED_DATA_DIR}/students_processed.csv", index=False)
    internships_df.to_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv", index=False)
    feedback_df.to_csv(f"{config.PROCESSED_DATA_DIR}/feedback_processed.csv", index=False)
    user_item_matrix.to_csv(f"{config.PROCESSED_DATA_DIR}/user_item_matrix.csv")
    
    print("✅ All processed data saved successfully!")
