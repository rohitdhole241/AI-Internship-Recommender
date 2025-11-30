"""
Visualize recommendation results and model performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')

from components.recommender import HybridRecommender
import config

def plot_recommendation_scores(student_id='S001'):
    """
    Plot recommendation scores for a student
    Args:
        student_id: Student ID to visualize recommendations for
    """
    # Initialize recommender
    recommender = HybridRecommender()
    
    # Get recommendations
    recommendations = recommender.recommend(student_id, top_n=10)
    
    # Extract data
    companies = [rec['company'] for rec in recommendations]
    scores = [rec['match_score'] for rec in recommendations]
    ratings = [rec['rating'] for rec in recommendations]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Match Scores
    bars1 = ax1.barh(companies, scores, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Match Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Company', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 10 Recommendations for Student {student_id}', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars1, scores)):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}',
                va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Match Score vs Company Rating
    colors = plt.cm.viridis(np.linspace(0, 1, len(companies)))
    scatter = ax2.scatter(scores, ratings, s=200, c=colors, alpha=0.7, edgecolors='black')
    
    ax2.set_xlabel('Match Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Company Rating (⭐)', fontsize=12, fontweight='bold')
    ax2.set_title('Match Score vs Company Rating', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(3.5, 5.0)
    
    # Add company labels to scatter points
    for i, company in enumerate(companies):
        ax2.annotate(company, (scores[i], ratings[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/recommendations_{student_id}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: visualizations/recommendations_{student_id}.png")
    plt.show()

def plot_model_comparison():
    """
    Compare content-based vs collaborative scores
    """
    recommender = HybridRecommender()
    
    # Get recommendations with detailed scores
    student_id = 'S001'
    students_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/students_processed.csv")
    student = students_df[students_df['student_id'] == student_id]
    student_profile = student['skill_profile'].values[0]
    
    internships_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv")
    
    # Get content scores
    content_recs = recommender.content_model.get_recommendations(student_profile, top_n=10)
    
    companies = content_recs['company'].values
    content_scores = content_recs['content_score'].values
    
    # Get collaborative scores
    collaborative_scores = []
    for internship_id in content_recs['internship_id']:
        collab_score = recommender.collaborative_model.predict(student_id, internship_id) / 5.0
        collaborative_scores.append(collab_score)
    
    # Create grouped bar chart
    x = np.arange(len(companies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, content_scores, width, label='Content-Based', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, collaborative_scores, width, label='Collaborative', color='skyblue', alpha=0.8)
    
    ax.set_xlabel('Company', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Content-Based vs Collaborative Filtering Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(companies, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/model_comparison.png")
    plt.show()

def plot_skill_domain_heatmap():
    """
    Create heatmap of skills vs domains
    """
    internships_df = pd.read_csv(f"{config.PROCESSED_DATA_DIR}/internships_processed.csv")
    
    # Extract top domains and skills
    domains = internships_df['domain'].value_counts().head(8).index
    
    # Count skill occurrences per domain
    skills_dict = {}
    top_skills = ['Python', 'Machine Learning', 'Java', 'JavaScript', 'React', 'SQL', 'Deep Learning', 'Data']
    
    for domain in domains:
        skills_dict[domain] = []
        domain_internships = internships_df[internships_df['domain'] == domain]
        
        for skill in top_skills:
            count = domain_internships['required_skills'].str.contains(skill, case=False, na=False).sum()
            skills_dict[domain].append(count)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data = np.array([skills_dict[domain] for domain in domains])
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(top_skills)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(top_skills, rotation=45, ha='right')
    ax.set_yticklabels(domains)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Internships', rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotations
    for i in range(len(domains)):
        for j in range(len(top_skills)):
            text = ax.text(j, i, data[i, j],
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Skills Required by Domain (Heatmap)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/skill_domain_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/skill_domain_heatmap.png")
    plt.show()

def main():
    """Generate all visualizations"""
    print("=" * 70)
    print("GENERATING RECOMMENDATION VISUALIZATIONS")
    print("=" * 70)
    
    print("\n📊 Creating recommendation scores plot...")
    plot_recommendation_scores('S001')
    
    print("\n📊 Creating model comparison plot...")
    plot_model_comparison()
    
    print("\n📊 Creating skill-domain heatmap...")
    plot_skill_domain_heatmap()
    
    print("\n" + "=" * 70)
    print("✅ All visualizations generated successfully!")
    print("=" * 70)
    print("\nSaved files:")
    print("  - visualizations/recommendations_S001.png")
    print("  - visualizations/model_comparison.png")
    print("  - visualizations/skill_domain_heatmap.png")

if __name__ == "__main__":
    main()
