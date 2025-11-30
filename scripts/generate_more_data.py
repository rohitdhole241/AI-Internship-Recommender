
"""
Generate larger synthetic dataset for better AI training
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration
NUM_STUDENTS = 100
NUM_INTERNSHIPS = 75
NUM_FEEDBACK = 300

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load existing data as templates
students_template = pd.read_csv('data/raw/students.csv')
internships_template = pd.read_csv('data/raw/internships.csv')
feedback_template = pd.read_csv('data/raw/feedback.csv')

# ========== Generate Students ==========
print("Generating students...")

student_names = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Dev", "Kiaan", "Arnav", "Krishna",
    "Diya", "Ananya", "Aadhya", "Saanvi", "Anika", "Myra", "Pari", "Sara", "Navya", "Ira",
    "Rohan", "Riya", "Kabir", "Ishaan", "Ayaan", "Meera", "Tara", "Aanya", "Zara", "Naina",
    "Laksh", "Advait", "Rudra", "Pranav", "Karthik", "Prisha", "Avni", "Ahaan", "Reyansh", "Veer"
]

surnames = ["Sharma", "Patel", "Kumar", "Singh", "Reddy", "Nair", "Iyer", "Gupta", "Joshi", "Desai",
           "Verma", "Mehta", "Bansal", "Agarwal", "Kapoor", "Malhotra", "Pillai", "Rao", "Menon", "Choudhary"]

branches = ["IT", "CSE", "ECE", "EEE"]
years = [3, 4]

skill_sets = [
    "Python,Machine Learning,SQL,Data Analysis",
    "Java,Spring Boot,MySQL,REST API",
    "React,JavaScript,HTML,CSS,Node.js",
    "Python,Deep Learning,TensorFlow,Computer Vision",
    "Flutter,Dart,Firebase,REST API",
    "Python,Data Visualization,Tableau,PowerBI",
    "C++,Algorithms,Data Structures,Problem Solving",
    "HTML,CSS,JavaScript,UI/UX Design,Figma",
    "Python,NLP,Machine Learning,Text Mining",
    "Angular,TypeScript,Node.js,MongoDB",
    "Python,R,Statistics,Machine Learning",
    "Android,Kotlin,Java,SQLite",
    "Python,IoT,Embedded Systems,Arduino",
    "React Native,JavaScript,Redux,API Integration",
    "PHP,Laravel,MySQL,Web Development",
    "Django,Python,PostgreSQL,REST API",
    "Vue.js,JavaScript,HTML,CSS",
    "Docker,Kubernetes,DevOps,CI/CD",
    "Python,Signal Processing,MATLAB,ML"
]

domain_interests = [
    "Data Science", "Backend Development", "Full Stack Development", "AI/ML",
    "Mobile Development", "Data Analytics", "Competitive Programming", "Frontend Development",
    "Natural Language Processing", "Web Development", "IoT Development", "DevOps", "Signal Processing"
]

locations = ["Bangalore", "Hyderabad", "Mumbai", "Pune", "Chennai", "Delhi", "Noida", "Gurgaon", "Kochi"]

past_internships = ["Google", "Microsoft", "Amazon", "TCS", "Infosys", "Wipro", "Accenture", "IBM", "Cognizant", "None"]

students_data = []

for i in range(1, NUM_STUDENTS + 1):
    student_id = f"S{i:03d}"
    name = f"{random.choice(student_names)} {random.choice(surnames)}"
    branch = random.choice(branches)
    year = random.choice(years)
    cgpa = round(random.uniform(7.0, 9.5), 1)
    skills = random.choice(skill_sets)
    domain = random.choice(domain_interests)
    location_pref = f"{random.choice(locations)},{random.choice(locations)}"
    past = random.choice(past_internships)
    
    students_data.append({
        'student_id': student_id,
        'name': name,
        'branch': branch,
        'year': year,
        'cgpa': cgpa,
        'skills': skills,
        'domain_interest': domain,
        'location_preference': location_pref,
        'past_internships': past
    })

students_df = pd.DataFrame(students_data)
print(f"✅ Generated {len(students_df)} students")

# ========== Generate Internships ==========
print("Generating internships...")

companies = [
    "Google", "Microsoft", "Amazon", "Flipkart", "TCS", "Infosys", "Wipro", "Accenture",
    "IBM", "Cognizant", "PhonePe", "Zomato", "Paytm", "NVIDIA", "HCL", "Bosch", "Qualcomm",
    "Swiggy", "Ola", "Razorpay", "Freshworks", "Byju's", "Myntra", "Urban Company", "Salesforce",
    "Adobe", "Oracle", "SAP", "Cisco", "Intel", "Samsung", "LG", "Walmart Labs", "LinkedIn",
    "Meta", "Apple", "Netflix", "Uber", "Airbnb", "Twitter"
]

roles = [
    "Data Science Intern", "Backend Developer Intern", "Full Stack Developer Intern",
    "AI Research Intern", "Flutter Developer Intern", "Business Analyst Intern",
    "Frontend Developer Intern", "NLP Intern", "Java Developer Intern", "React Native Developer",
    "Backend Engineer Intern", "UI Developer Intern", "Deep Learning Intern", "PHP Developer Intern",
    "IoT Engineer Intern", "Signal Processing Intern", "ML Engineer Intern", "Android Developer Intern",
    "Software Engineer Intern", "DevOps Intern"
]

internships_data = []

for i in range(1, NUM_INTERNSHIPS + 1):
    internship_id = f"I{i:03d}"
    company = random.choice(companies)
    domain = random.choice(domain_interests)
    role = random.choice(roles)
    required_skills = random.choice(skill_sets)
    location = random.choice(locations)
    duration = random.choice([3, 4, 5, 6])
    stipend = random.choice([22000, 25000, 28000, 30000, 35000, 38000, 40000, 42000, 45000, 48000, 50000, 55000, 60000])
    rating = round(random.uniform(3.8, 5.0), 1)
    reviews = random.randint(30, 200)
    description = f"Work on real-world {domain} projects with cutting-edge technologies"
    
    internships_data.append({
        'internship_id': internship_id,
        'company': company,
        'domain': domain,
        'role': role,
        'required_skills': required_skills,
        'location': location,
        'duration_months': duration,
        'stipend': stipend,
        'rating': rating,
        'total_reviews': reviews,
        'description': description
    })

internships_df = pd.DataFrame(internships_data)
print(f"✅ Generated {len(internships_df)} internships")

# ========== Generate Feedback ==========
print("Generating feedback...")

feedback_texts = [
    "Excellent learning experience with great mentorship",
    "Good hands-on projects and team collaboration",
    "Amazing work culture and challenging projects",
    "Great exposure to industry-standard tools",
    "Solid internship with real-world impact",
    "Best internship for learning cutting-edge tech",
    "Good experience but could improve code reviews",
    "Fantastic team and meaningful work",
    "Great stipend and learning opportunities",
    "Excellent mentors and project ownership"
]

recommendations = ["Yes", "Yes", "Yes", "Yes", "Maybe", "No"]

feedback_data = []

# Generate realistic feedback (not all students rate all internships)
for _ in range(NUM_FEEDBACK):
    student_id = f"S{random.randint(1, NUM_STUDENTS):03d}"
    internship_id = f"I{random.randint(1, NUM_INTERNSHIPS):03d}"
    rating = random.choice([3, 4, 4, 4, 5, 5, 5])  # Bias towards positive ratings
    feedback_text = random.choice(feedback_texts)
    
    # Random date in last 6 months
    days_ago = random.randint(0, 180)
    completion_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    would_recommend = random.choice(recommendations)
    
    # Avoid duplicate student-internship pairs
    if not any(f['student_id'] == student_id and f['internship_id'] == internship_id for f in feedback_data):
        feedback_data.append({
            'student_id': student_id,
            'internship_id': internship_id,
            'rating': rating,
            'feedback_text': feedback_text,
            'completion_date': completion_date,
            'would_recommend': would_recommend
        })

feedback_df = pd.DataFrame(feedback_data)
print(f"✅ Generated {len(feedback_df)} feedback entries")

# ========== Save to CSV ==========
print("\nSaving to CSV files...")

students_df.to_csv('data/raw/students.csv', index=False)
internships_df.to_csv('data/raw/internships.csv', index=False)
feedback_df.to_csv('data/raw/feedback.csv', index=False)

print("\n" + "="*70)
print("✅ DATASET EXPANSION COMPLETE!")
print("="*70)
print(f"Students: {len(students_df)} records")
print(f"Internships: {len(internships_df)} records")
print(f"Feedback: {len(feedback_df)} records")
print(f"Total data points: {len(students_df) + len(internships_df) + len(feedback_df)}")
print("\nSparsity reduced! AI models will now learn better patterns.")
print("="*70)
