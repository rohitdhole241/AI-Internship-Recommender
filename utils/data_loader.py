import pandas as pd
import config

def load_students_data():
    """Load students dataset"""
    try:
        df = pd.read_csv(config.STUDENTS_FILE)
        print(f"✅ Loaded {len(df)} students")
        return df
    except Exception as e:
        print(f"❌ Error loading students data: {e}")
        return None

def load_internships_data():
    """Load internships dataset"""
    try:
        df = pd.read_csv(config.INTERNSHIPS_FILE)
        print(f"✅ Loaded {len(df)} internships")
        return df
    except Exception as e:
        print(f"❌ Error loading internships data: {e}")
        return None

def load_feedback_data():
    """Load feedback dataset"""
    try:
        df = pd.read_csv(config.FEEDBACK_FILE)
        print(f"✅ Loaded {len(df)} feedback entries")
        return df
    except Exception as e:
        print(f"❌ Error loading feedback data: {e}")
        return None

def load_all_data():
    """Load all datasets at once"""
    students = load_students_data()
    internships = load_internships_data()
    feedback = load_feedback_data()
    return students, internships, feedback
