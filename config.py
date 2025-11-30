import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# CSV file paths
STUDENTS_FILE = os.path.join(RAW_DATA_DIR, 'students.csv')
INTERNSHIPS_FILE = os.path.join(RAW_DATA_DIR, 'internships.csv')
FEEDBACK_FILE = os.path.join(RAW_DATA_DIR, 'feedback.csv')

# Model parameters
CONTENT_WEIGHT = 0.6  # Alpha for hybrid model
COLLABORATIVE_WEIGHT = 0.4  # (1 - Alpha)
TOP_N_RECOMMENDATIONS = 5

# Collaborative filtering parameters
MIN_RATING = 1
MAX_RATING = 5
