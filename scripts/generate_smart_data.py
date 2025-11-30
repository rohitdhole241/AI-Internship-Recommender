"""
Generate SMART synthetic dataset with logical skill-internship matching
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

NUM_STUDENTS = 100
NUM_INTERNSHIPS = 75
NUM_FEEDBACK = 350

# ========== SMART SKILL-DOMAIN MAPPING ==========

# Define skill groups that naturally match domains
SKILL_DOMAIN_MAP = {
    'Data Science': {
        'skills': ["Python,Machine Learning,SQL,Data Analysis,Pandas,NumPy",
                   "Python,Machine Learning,Statistics,SQL,Scikit-learn",
                   "Python,R,Statistics,Data Visualization,Machine Learning"],
        'internship_skills': ["Python,Machine Learning,SQL,Statistics",
                              "Python,Data Analysis,Pandas,Scikit-learn",
                              "R,Statistics,Data Visualization,ML"]
    },
    'AI/ML': {
        'skills': ["Python,Deep Learning,TensorFlow,Computer Vision,Neural Networks",
                   "Python,Deep Learning,PyTorch,NLP,Transformers",
                   "Python,Machine Learning,Deep Learning,Keras,CNN"],
        'internship_skills': ["Python,Deep Learning,TensorFlow,Computer Vision",
                              "Python,PyTorch,Neural Networks,NLP",
                              "Deep Learning,Keras,Machine Learning,AI"]
    },
    'Backend Development': {
        'skills': ["Java,Spring Boot,MySQL,REST API,Microservices",
                   "Python,Django,PostgreSQL,REST API,Flask",
                   "Node.js,Express,MongoDB,REST API,JavaScript"],
        'internship_skills': ["Java,Spring Boot,MySQL,REST API",
                              "Python,Django,PostgreSQL,API Development",
                              "Node.js,Express,MongoDB,Backend"]
    },
    'Frontend Development': {
        'skills': ["React,JavaScript,HTML,CSS,TypeScript",
                   "Vue.js,JavaScript,HTML,CSS,Vuex",
                   "Angular,TypeScript,HTML,CSS,RxJS"],
        'internship_skills': ["React,JavaScript,HTML,CSS",
                              "Vue.js,JavaScript,Frontend,UI",
                              "Angular,TypeScript,HTML,CSS"]
    },
    'Full Stack Development': {
        'skills': ["React,Node.js,MongoDB,JavaScript,Express",
                   "Angular,Node.js,MySQL,TypeScript,Express",
                   "Vue.js,Django,PostgreSQL,JavaScript,Python"],
        'internship_skills': ["React,Node.js,MongoDB,Full Stack",
                              "Angular,Node.js,MySQL,JavaScript",
                              "Vue.js,Python,Django,Full Stack"]
    },
    'Mobile Development': {
        'skills': ["Flutter,Dart,Firebase,REST API,Mobile",
                   "React Native,JavaScript,Redux,Mobile,API",
                   "Android,Kotlin,Java,SQLite,Mobile"],
        'internship_skills': ["Flutter,Dart,Firebase,Mobile Development",
                              "React Native,JavaScript,Mobile,Redux",
                              "Android,Kotlin,Java,Mobile Apps"]
    }
}

# ========== Generate Students ==========
print("Generating students with smart skill matching...")

first_names = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Diya", "Ananya", "Saanvi", "Rohan", "Riya",
               "Kabir", "Ishaan", "Meera", "Tara", "Laksh", "Prisha", "Karthik", "Navya", "Dev", "Sara"]
last_names = ["Sharma", "Patel", "Kumar", "Singh", "Reddy", "Nair", "Iyer", "Gupta", "Joshi", "Desai"]

students_data = []
domain_distribution = list(SKILL_DOMAIN_MAP.keys())

for i in range(1, NUM_STUDENTS + 1):
    # Assign domain and matching skills
    domain = random.choice(domain_distribution)
    skills = random.choice(SKILL_DOMAIN_MAP[domain]['skills'])
    
    students_data.append({
        'student_id': f"S{i:03d}",
        'name': f"{random.choice(first_names)} {random.choice(last_names)}",
        'branch': random.choice(["IT", "CSE", "ECE"]),
        'year': random.choice([3, 4]),
        'cgpa': round(random.uniform(7.0, 9.5), 1),
        'skills': skills,
        'domain_interest': domain,
        'location_preference': f"{random.choice(['Bangalore', 'Hyderabad', 'Mumbai'])},{random.choice(['Pune', 'Chennai', 'Delhi'])}",
        'past_internships': random.choice(["Google", "Microsoft", "Amazon", "TCS", "None"])
    })

students_df = pd.DataFrame(students_data)
print(f"✅ Generated {len(students_df)} students")

# ========== Generate Internships ==========
print("Generating internships with matching skill requirements...")

companies = ["Google", "Microsoft", "Amazon", "Flipkart", "IBM", "NVIDIA", "Adobe", "Salesforce",
             "PhonePe", "Zomato", "Razorpay", "Freshworks", "Myntra", "Swiggy", "Ola", "Paytm"]

roles_map = {
    'Data Science': ["Data Science Intern", "Data Analyst Intern", "ML Engineer Intern"],
    'AI/ML': ["AI Research Intern", "Deep Learning Intern", "ML Intern"],
    'Backend Development': ["Backend Developer Intern", "Backend Engineer Intern", "API Developer Intern"],
    'Frontend Development': ["Frontend Developer Intern", "UI Developer Intern", "React Developer Intern"],
    'Full Stack Development': ["Full Stack Developer Intern", "Software Engineer Intern", "Web Developer Intern"],
    'Mobile Development': ["Mobile Developer Intern", "Flutter Developer Intern", "Android Developer Intern"]
}

internships_data = []

for i in range(1, NUM_INTERNSHIPS + 1):
    domain = random.choice(domain_distribution)
    required_skills = random.choice(SKILL_DOMAIN_MAP[domain]['internship_skills'])
    
    internships_data.append({
        'internship_id': f"I{i:03d}",
        'company': random.choice(companies),
        'domain': domain,
        'role': random.choice(roles_map[domain]),
        'required_skills': required_skills,
        'location': random.choice(['Bangalore', 'Hyderabad', 'Mumbai', 'Pune', 'Chennai']),
        'duration_months': random.choice([3, 4, 5, 6]),
        'stipend': random.choice([25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]),
        'rating': round(random.uniform(4.0, 5.0), 1),
        'total_reviews': random.randint(40, 180),
        'description': f"Work on cutting-edge {domain} projects"
    })

internships_df = pd.DataFrame(internships_data)
print(f"✅ Generated {len(internships_df)} internships")

# ========== Generate SMART Feedback ==========
print("Generating smart feedback with domain-matched ratings...")

feedback_data = []

for _ in range(NUM_FEEDBACK):
    student_id = f"S{random.randint(1, NUM_STUDENTS):03d}"
    internship_id = f"I{random.randint(1, NUM_INTERNSHIPS):03d}"
    
    # Get student and internship domains
    student_domain = students_df[students_df['student_id'] == student_id]['domain_interest'].values[0]
    internship_domain = internships_df[internships_df['internship_id'] == internship_id]['domain'].values[0]
    
    # If domains match, give higher ratings (smart feedback!)
    if student_domain == internship_domain:
        rating = random.choice([4, 4, 5, 5, 5])  # High ratings for matched domains
        would_recommend = random.choice(['Yes', 'Yes', 'Yes', 'Maybe'])
    else:
        rating = random.choice([2, 3, 3, 4])  # Lower ratings for mismatched domains
        would_recommend = random.choice(['Maybe', 'Maybe', 'No', 'Yes'])
    
    # Avoid duplicates
    if not any(f['student_id'] == student_id and f['internship_id'] == internship_id for f in feedback_data):
        feedback_data.append({
            'student_id': student_id,
            'internship_id': internship_id,
            'rating': rating,
            'feedback_text': "Great internship experience!" if rating >= 4 else "Good learning opportunity",
            'completion_date': (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d'),
            'would_recommend': would_recommend
        })

feedback_df = pd.DataFrame(feedback_data)
print(f"✅ Generated {len(feedback_df)} smart feedback entries")

# ========== Save ==========
students_df.to_csv('data/raw/students.csv', index=False)
internships_df.to_csv('data/raw/internships.csv', index=False)
feedback_df.to_csv('data/raw/feedback.csv', index=False)

print("\n" + "="*70)
print("✅ SMART DATASET GENERATION COMPLETE!")
print("="*70)
print(f"Students: {len(students_df)} (with domain-matched skills)")
print(f"Internships: {len(internships_df)} (with matching requirements)")
print(f"Feedback: {len(feedback_df)} (with intelligent rating patterns)")
print("\nNow your AI will show MUCH BETTER match scores!")
print("="*70)
