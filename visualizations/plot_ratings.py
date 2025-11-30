"""
Script to visualize rating distribution and internship statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from utils.data_loader import load_all_data

def plot_rating_distribution(feedback):
    """Plot rating distribution"""
    plt.figure(figsize=(10, 6))
    
    rating_counts = feedback['rating'].value_counts().sort_index()
    bars = plt.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='navy')
    
    plt.title('Internship Rating Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Rating (Stars)', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/rating_distribution.png', dpi=300)
    print("✅ Saved: visualizations/rating_distribution.png")
    plt.show()

def plot_top_companies(internships):
    """Plot top companies by average rating"""
    plt.figure(figsize=(12, 6))
    
    company_ratings = internships.groupby('company')['rating'].mean().sort_values(ascending=False).head(10)
    
    bars = plt.barh(company_ratings.index, company_ratings.values, color='coral')
    plt.title('Top 10 Companies by Average Rating', fontsize=16, fontweight='bold')
    plt.xlabel('Average Rating', fontsize=12)
    plt.ylabel('Company', fontsize=12)
    plt.xlim(0, 5)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/top_companies.png', dpi=300)
    print("✅ Saved: visualizations/top_companies.png")
    plt.show()

def main():
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    students, internships, feedback = load_all_data()
    
    print("\n📊 Creating rating distribution plot...")
    plot_rating_distribution(feedback)
    
    print("\n📊 Creating top companies plot...")
    plot_top_companies(internships)
    
    print("\n✅ All visualizations generated!")

if __name__ == "__main__":
    main()
