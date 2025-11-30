AI-Based Internship Recommendation System

This project is an AI-powered hybrid recommendation system that suggests the most suitable internships to students based on their skills, interests, academic profile, and feedback patterns.
The system integrates NLP, Machine Learning, and Recommender System algorithms to deliver personalized internship recommendations through a Streamlit web application.

Features

Personalized internship recommendations

Hybrid model combining content-based + collaborative filtering

NLP-based skill matching using TF-IDF and Cosine Similarity

Learns from student feedback and peer behavior

Displays match score, stipend, duration, and company rating

User-friendly interface with Streamlit

Tech Stack
Component	Technology
Language	Python
Framework	Streamlit
ML Libraries	Scikit-Learn, NumPy, Pandas
NLP	TF-IDF Vectorizer + Cosine Similarity
Recommender System	Content-Based, Collaborative, Hybrid
Visualization	Matplotlib / Seaborn (optional)
📌 System Workflow
Dataset → Preprocessing → ML Models → Hybrid Recommender → Streamlit UI

Models Used
Model	Description
Content-Based Filtering	Matches student skills to internship requirements
Collaborative Filtering	Predicts based on similar students’ preferences
Hybrid Model	Weighted combination of both models for best accuracy
Dataset Structure
File	Purpose
students.csv	Student profile, CGPA, skills, and domain interest
internships.csv	Internship role, required skills, company & stipend
feedback.csv	Ratings and recommendations given by students

Processed versions of these datasets are used for model training.

How to Run the Project
 Install Dependencies
pip install -r requirements.txt

▶ Run Streamlit Application
streamlit run app.py

▶ Run Recommender from Terminal (optional)
python main.py

 Future Enhancements

Support for brand-new users without existing data

Resume parsing for auto-extracting skills

Admin dashboard for companies to post internships

Push notifications / email alerts for new opportunities

Developed By

Rohit Dhole
