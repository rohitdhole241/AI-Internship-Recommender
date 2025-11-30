"""
Streamlit Web Application for AI-Based Internship Recommendation System
"""

import streamlit as st
import pandas as pd
import sys
sys.path.append('.')

from components.recommender import HybridRecommender
import config

# Page configuration
st.set_page_config(
    page_title="AI Internship Recommender",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'recommender' not in st.session_state:
    with st.spinner("Loading AI models..."):
        st.session_state.recommender = HybridRecommender()
        st.session_state.students_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/students_processed.csv")

# Title
st.title("🎯 AI-Based Hybrid Recommendation System")
st.subheader("Intelligent Internship Recommendations for Students")

# Sidebar
st.sidebar.header("📊 System Information")
st.sidebar.info(
    """
    **Algorithms Used:**
    - Content-Based Filtering (TF-IDF)
    - Collaborative Filtering (User-based)
    - Hybrid Model (Weighted Combination)
    
    **Features:**
    - Skill Matching
    - Domain Alignment
    - Peer Feedback Analysis
    """
)

# Main content
st.markdown("---")

# Student selection
col1, col2 = st.columns([2, 1])

with col1:
    student_ids = st.session_state.students_df['student_id'].tolist()
    selected_student = st.selectbox(
        "Select Student ID:",
        student_ids,
        index=0
    )

with col2:
    num_recommendations = st.slider(
        "Number of Recommendations:",
        min_value=3,
        max_value=10,
        value=5
    )

# Display student info
if selected_student:
    student_info = st.session_state.students_df[
        st.session_state.students_df['student_id'] == selected_student
    ].iloc[0]
    
    st.markdown("### 👤 Student Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Name", student_info['name'])
    with col2:
        st.metric("Branch", student_info['branch'])
    with col3:
        st.metric("Year", student_info['year'])
    with col4:
        st.metric("CGPA", f"{student_info['cgpa']}/10")
    
    st.markdown(f"**Skills:** {student_info['skills']}")
    st.markdown(f"**Domain Interest:** {student_info['domain_interest']}")

# Generate recommendations button
st.markdown("---")

if st.button("🚀 Generate Recommendations", type="primary"):
    with st.spinner("Generating personalized recommendations..."):
        recommendations = st.session_state.recommender.recommend(
            selected_student,
            top_n=num_recommendations
        )
    
    st.success(f"✅ Generated {len(recommendations)} recommendations!")
    
    # Display recommendations
    st.markdown("### 🎯 Recommended Internships")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"#{i} - {rec['company']} - {rec['role']}", expanded=(i <= 3)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Domain:** {rec['domain']}")
                st.markdown(f"**Location:** {rec['location']}")
                st.markdown(f"**Duration:** {rec['duration_months']} months")
                
            with col2:
                st.metric("Match Score", f"{rec['match_score']:.2%}")
                st.metric("Company Rating", f"{rec['rating']}⭐")
                st.metric("Stipend", f"₹{rec['stipend']:,}")
            
            # Progress bar for match score
            st.progress(rec['match_score'])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | AI & Machine Learning Mini Project</p>
    </div>
    """,
    unsafe_allow_html=True
)
