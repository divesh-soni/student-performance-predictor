from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Model information (to display on dashboard)
MODEL_INFO = {
    'type': 'Linear Regression',
    'r2_score': 0.9012,
    'mae': 4.41,
    'rmse': 5.46,
    'features': ['Study Hours', 'Sleep Hours', 'Social Media', 'Netflix', 'Exercise', 'Mental Health', 'Age'],
    'feature_importance': {
        'Study Hours/Day': 13.84,
        'Mental Health Rating': 5.55,
        'Exercise Frequency': 2.90,
        'Sleep Hours': 2.44,
        'Netflix Hours': -2.50,
        'Social Media Hours': -3.04,
        'Age': -0.05
    }
}

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "student_model.pkl"
)

# Load model with error handling
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully:", type(pipeline))
except FileNotFoundError:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please train the model using preprocessing_02.ipynb first.")
    pipeline = None
except (EOFError, ValueError) as e:
    print(f"ERROR loading model (corrupted file): {e}")
    pipeline = None
except Exception as e:
    print(f"ERROR loading model: {type(e).__name__}: {e}")
    pipeline = None

# Feature names - MUST MATCH your training features
FEATURE_NAMES = [
    'age', 'study_hours_per_day', 'social_media_hours',
    'netflix_hours', 'sleep_hours', 'exercise_frequency',
    'mental_health_rating'
]

# Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_name TEXT,
                  age INTEGER,
                  study_hours REAL,
                  social_media_hours REAL,
                  netflix_hours REAL,
                  sleep_hours REAL,
                  exercise_frequency INTEGER,
                  mental_health_rating INTEGER,
                  predicted_score REAL,
                  performance_category TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()


def save_prediction(data, score, category):
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (student_name, age, study_hours, social_media_hours, netflix_hours,
                      sleep_hours, exercise_frequency, mental_health_rating, 
                      predicted_score, performance_category, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data['student_name'], int(data['age']), float(data['study_hours']),
                   float(data['social_hours']), float(data['netflix_hours']),
                   float(data['sleep_hours']), int(data['exercise_freq']),
                   int(data['mental_health']), score, category,
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        logger.info(f"Saved prediction for {data['student_name']}")
    except sqlite3.OperationalError as e:
        logger.error(f"Database error saving prediction: {e}")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Data validation error saving prediction: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving prediction: {type(e).__name__}: {e}")


def get_all_predictions():
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
        rows = c.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'name': row[1],
                'age': row[2],
                'study_hours': row[3],
                'social_hours': row[4],
                'netflix_hours': row[5],
                'sleep_hours': row[6],
                'exercise_freq': row[7],
                'mental_health': row[8],
                'score': round(row[9], 2),
                'category': row[10],
                'date': row[11]
            })
        return history
    except sqlite3.OperationalError as e:
        logger.error(f"Database error fetching predictions: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching predictions: {type(e).__name__}: {e}")
        return []


def delete_prediction(pred_id):
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        logger.error(f"Database error deleting prediction: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting prediction: {type(e).__name__}: {e}")
        return False


def get_dataset_stats():
    """Calculate real statistics from the dataset"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)

        stats = {
            'total_students': len(df),
            'avg_score': round(df['exam_score'].mean(), 1),
            'highest_score': df['exam_score'].max(),
            'lowest_score': df['exam_score'].min(),
            'avg_study': round(df['study_hours_per_day'].mean(), 1),
            'avg_sleep': round(df['sleep_hours'].mean(), 1),
            'avg_social': round(df['social_media_hours'].mean(), 1),
            'avg_netflix': round(df['netflix_hours'].mean(), 1),
            'avg_exercise': round(df['exercise_frequency'].mean(), 1),
            'avg_mental': round(df['mental_health_rating'].mean(), 1),
            'pass_rate': round((df['exam_score'] >= 50).sum() / len(df) * 100, 1),
            'excellent_rate': round((df['exam_score'] >= 85).sum() / len(df) * 100, 1),
        }
        return stats
    except FileNotFoundError as e:
        logger.error(f"Dataset file error: {e}")
        return None
    except (pd.errors.ParserError, KeyError) as e:
        logger.error(f"Data parsing error calculating stats: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating stats: {type(e).__name__}: {e}")
        return None


def get_average_stats():
    """Calculate average values from the dataset for comparison"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)

        stats = {
            'avg_score': round(df['exam_score'].mean(), 1),
            'avg_study': round(df['study_hours_per_day'].mean(), 1),
            'avg_sleep': round(df['sleep_hours'].mean(), 1),
            'avg_social': round(df['social_media_hours'].mean(), 1),
            'avg_exercise': round(df['exercise_frequency'].mean(), 1),
            'avg_mental': round(df['mental_health_rating'].mean(), 1),
            'avg_netflix': round(df['netflix_hours'].mean(), 1),
        }
        return stats
    except Exception as e:
        logger.error(f"Error calculating average stats: {e}")
        # Return fallback values based on real dataset
        return {
            'avg_score': 69.6, 'avg_study': 3.6, 'avg_sleep': 6.5,
            'avg_social': 2.5, 'avg_exercise': 3.0, 'avg_mental': 5.4, 'avg_netflix': 1.8
        }


def get_letter_grade(score):
    if score >= 90:
        return 'A', 'Excellent!'
    elif score >= 80:
        return 'B', 'Very Good!'
    elif score >= 70:
        return 'C', 'Good'
    elif score >= 60:
        return 'D', 'Need Improvement'
    else:
        return 'F', 'Needs Significant Improvement'


def validate_input(data):
    errors = []

    if not data.get('student_name', '').strip():
        errors.append("Student name is required")
    elif len(data.get('student_name', '').strip()) < 2:
        errors.append("Please enter a valid name (at least 2 characters)")

    try:
        age = int(data['age'])
        if age < 15 or age > 100:
            errors.append("Age must be between 15 and 100")
    except (ValueError, TypeError):
        errors.append("Please enter a valid age")

    try:
        study = float(data['study_hours'])
        if study < 0 or study > 24:
            errors.append("Study hours must be between 0 and 24")
    except (ValueError, TypeError):
        errors.append("Please enter valid study hours")

    try:
        social = float(data['social_hours'])
        if social < 0 or social > 24:
            errors.append("Social media hours must be between 0 and 24")
    except (ValueError, TypeError):
        errors.append("Please enter valid social media hours")

    try:
        netflix = float(data['netflix_hours'])
        if netflix < 0 or netflix > 24:
            errors.append("Entertainment hours must be between 0 and 24")
    except (ValueError, TypeError):
        errors.append("Please enter valid entertainment hours")

    try:
        sleep = float(data['sleep_hours'])
        if sleep < 3 or sleep > 14:
            errors.append("Sleep hours should be between 3 and 14")
    except (ValueError, TypeError):
        errors.append("Please enter valid sleep hours")

    try:
        exercise = int(data['exercise_freq'])
        if exercise < 0 or exercise > 7:
            errors.append("Exercise frequency must be between 0 and 7 days per week")
    except (ValueError, TypeError):
        errors.append("Please enter valid exercise frequency")

    try:
        mental = int(data['mental_health'])
        if mental < 1 or mental > 10:
            errors.append("Mental health rating must be between 1 and 10")
    except (ValueError, TypeError):
        errors.append("Please enter valid mental health rating")

    return errors


def get_performance_category(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"


def generate_strengths_weaknesses(features):
    strengths = []
    weaknesses = []

    # Study hours
    if features['study_hours_per_day'] >= 5:
        strengths.append("Strong study habits - you study 5+ hours daily")
    elif features['study_hours_per_day'] >= 4:
        strengths.append("Good study routine - consistent daily studying")
    elif features['study_hours_per_day'] < 3:
        weaknesses.append("Low study hours - consider increasing to 4-5 hours daily")

    # Social media
    if features['social_media_hours'] <= 2:
        strengths.append("Excellent digital discipline - low social media usage")
    elif features['social_media_hours'] <= 3:
        strengths.append("Moderate social media usage")
    elif features['social_media_hours'] > 4:
        weaknesses.append("High social media usage - may affect study focus")

    # Sleep
    if 7 <= features['sleep_hours'] <= 9:
        strengths.append("Optimal sleep schedule - great for memory consolidation")
    elif features['sleep_hours'] < 6:
        weaknesses.append("Insufficient sleep - may impact concentration and learning")
    elif features['sleep_hours'] > 9:
        weaknesses.append("Excessive sleep - might indicate low energy levels")

    # Exercise
    if features['exercise_frequency'] >= 3:
        strengths.append("Regular exercise - boosts brain function and mental health")
    elif features['exercise_frequency'] >= 1:
        strengths.append("Some physical activity - consider increasing to 3+ days/week")
    else:
        weaknesses.append("No regular exercise - physical activity improves cognitive function")

    # Mental health
    if features['mental_health_rating'] >= 8:
        strengths.append("Excellent mental well-being - key for academic success")
    elif features['mental_health_rating'] >= 6:
        strengths.append("Good mental health - keep maintaining it")
    elif features['mental_health_rating'] < 5:
        weaknesses.append("Mental health concerns - consider speaking with a counselor")

    if not strengths:
        strengths.append("You have potential to improve with better daily habits")

    if not weaknesses:
        weaknesses.append(
            "Excellent performance across key lifestyle indicators. Continue maintaining consistency."
        )

    return strengths, weaknesses


def generate_recommendations(features, score):
    recommendations = []

    # Study hours recommendations
    if features['study_hours_per_day'] < 4:
        recommendations.append({
            'area': 'Study Habits',
            'suggestion': f'Increase study time from {features["study_hours_per_day"]} to 4-5 hours daily',
            'actionable_tip': 'Use Pomodoro technique: 25 min focused study, 5 min break'
        })
    elif features['study_hours_per_day'] > 7:
        recommendations.append({
            'area': 'Study-Life Balance',
            'suggestion': f'You study {features["study_hours_per_day"]} hours daily - ensure quality over quantity',
            'actionable_tip': 'Take regular breaks and ensure you are retaining information effectively'
        })

    # Social media recommendations
    if features['social_media_hours'] > 3:
        recommendations.append({
            'area': 'Digital Wellness',
            'suggestion': f'Reduce social media from {features["social_media_hours"]} to under 2 hours daily',
            'actionable_tip': 'Use app timers and schedule specific times for social media checks'
        })

    # Sleep recommendations
    if features['sleep_hours'] < 7:
        recommendations.append({
            'area': 'Sleep Hygiene',
            'suggestion': 'Aim for 7-8 hours of quality sleep',
            'actionable_tip': 'Maintain consistent bedtime and avoid screens 30 mins before sleep'
        })
    elif features['sleep_hours'] > 9:
        recommendations.append({
            'area': 'Sleep Schedule',
            'suggestion': 'Consider reducing sleep to 7-8 hours for optimal daytime energy',
            'actionable_tip': 'Try waking up at the same time every day to regulate your circadian rhythm'
        })

    # Exercise recommendations
    if features['exercise_frequency'] < 2:
        recommendations.append({
            'area': 'Physical Activity',
            'suggestion': 'Start with 2-3 exercise sessions per week',
            'actionable_tip': 'Even 15-20 minutes of walking or stretching daily makes a difference'
        })

    # Mental health recommendations
    if features['mental_health_rating'] < 5:
        recommendations.append({
            'area': 'Mental Wellness',
            'suggestion': 'Prioritize mental health support',
            'actionable_tip': 'Talk to a counselor, practice mindfulness, or join a student support group'
        })

    # Academic support for low scores
    if score < 50:
        recommendations.append({
            'area': 'Academic Support',
            'suggestion': 'Seek additional academic help',
            'actionable_tip': 'Join study groups, visit office hours, or get a tutor for difficult subjects'
        })

    if not recommendations:
        recommendations.append({
            'area': 'Keep Going',
            'suggestion': 'You are doing well! Maintain your good habits',
            'actionable_tip': 'Track your progress weekly to stay motivated and identify areas for improvement'
        })

    return recommendations


def calculate_percentile(score):
    """Calculate percentile based on dataset"""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)
        percentile = round((df['exam_score'] < score).sum() / len(df) * 100, 1)
        return percentile
    except Exception as e:
        logger.error(f"Error calculating percentile: {e}")
        return None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        student_data = {
            'student_name': request.form.get('student_name', ''),
            'age': request.form.get('age', ''),
            'study_hours': request.form.get('study_hours', ''),
            'social_hours': request.form.get('social_hours', ''),
            'netflix_hours': request.form.get('netflix_hours', ''),
            'sleep_hours': request.form.get('sleep_hours', ''),
            'exercise_freq': request.form.get('exercise_freq', ''),
            'mental_health': request.form.get('mental_health', '')
        }

        errors = validate_input(student_data)

        if errors:
            return render_template('predict.html', errors=errors, form_data=student_data)

        # Check if model is loaded
        if pipeline is None:
            errors = ["Model not loaded. Please ensure model is trained and available."]
            return render_template('predict.html', errors=errors, form_data=student_data)

        features = {
            'age': int(student_data['age']),
            'study_hours_per_day': float(student_data['study_hours']),
            'social_media_hours': float(student_data['social_hours']),
            'netflix_hours': float(student_data['netflix_hours']),
            'sleep_hours': float(student_data['sleep_hours']),
            'exercise_frequency': int(student_data['exercise_freq']),
            'mental_health_rating': int(student_data['mental_health'])
        }

        df_pred = pd.DataFrame([[
            features['age'], features['study_hours_per_day'],
            features['social_media_hours'], features['netflix_hours'],
            features['sleep_hours'], features['exercise_frequency'],
            features['mental_health_rating']
        ]], columns=FEATURE_NAMES)

        try:
            predicted_score = pipeline.predict(df_pred)[0]
            # FIX: Clamp score to valid range 0–100 and cast to plain float
            predicted_score = round(float(np.clip(predicted_score, 0, 100)), 2)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            errors = ["Error making prediction. Please try again."]
            return render_template('predict.html', errors=errors, form_data=student_data)

        category = get_performance_category(predicted_score)
        strengths, weaknesses = generate_strengths_weaknesses(features)
        recommendations = generate_recommendations(features, predicted_score)

        save_prediction(student_data, predicted_score, category)

        # FIX: Explicitly cast all numeric feature values to plain Python types
        # to ensure proper JSON serialisation inside Flask's session (cookie).
        safe_features = {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in features.items()
        }

        session['last_prediction'] = {
            'student_name': student_data['student_name'],
            'score': float(predicted_score),
            'category': category,
            'features': safe_features,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }

        return redirect(url_for('report'))

    return render_template('predict.html', errors=[], form_data={})


@app.route('/report')
def report():
    prediction = session.get('last_prediction')
    if not prediction:
        return redirect(url_for('predict'))

    # Get average stats from dataset
    avg_stats = get_average_stats()

    # Calculate percentile
    percentile = calculate_percentile(prediction['score'])
    if percentile is None:
        percentile = 50

    # Get letter grade
    letter_grade, grade_msg = get_letter_grade(prediction['score'])

    return render_template('report.html', prediction=prediction, avg_stats=avg_stats,
                           percentile=percentile, letter_grade=letter_grade, grade_msg=grade_msg)


@app.route('/history')
def history():
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)


@app.route('/dashboard')
def dashboard():
    stats = get_dataset_stats()

    # If stats couldn't be loaded, show fallback values
    if stats is None:
        stats = {
            'total_students': 1000,
            'avg_score': 69.6,
            'highest_score': 100,
            'lowest_score': 18.4,
            'avg_study': 3.6,
            'avg_sleep': 6.5,
            'avg_social': 2.5,
            'avg_netflix': 1.8,
            'avg_exercise': 3.0,
            'avg_mental': 5.4,
            'pass_rate': 75,
            'excellent_rate': 25,
            'error': True
        }

    return render_template('dashboard.html', stats=stats, model_info=MODEL_INFO)


@app.route('/about')
def about():
    return render_template('about.html', model_info=MODEL_INFO)


@app.route('/api/delete/<int:pred_id>', methods=['DELETE'])
def api_delete(pred_id):
    success = delete_prediction(pred_id)
    return jsonify({'success': success})


# API endpoint for score distribution chart
@app.route('/api/score-distribution')
def api_score_distribution():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Dataset not found'}), 404

        df = pd.read_csv(csv_path)

        # Create bins for scores
        bins = [0, 20, 40, 50, 60, 70, 80, 90, 100]
        labels = ['0-20', '21-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

        df['score_range'] = pd.cut(df['exam_score'], bins=bins, labels=labels, right=True)
        counts = df['score_range'].value_counts().reindex(labels).fillna(0).astype(int).tolist()

        return jsonify({'bins': labels, 'counts': counts})
    except Exception as e:
        logger.error(f"Error in score distribution: {e}")
        return jsonify({'error': str(e)}), 500


# API endpoint for category distribution
@app.route('/api/category-distribution')
def api_category_distribution():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Dataset not found'}), 404

        df = pd.read_csv(csv_path)

        def get_cat(score):
            if score >= 85:
                return 'Excellent'
            elif score >= 70:
                return 'Good'
            elif score >= 50:
                return 'Average'
            else:
                return 'Poor'

        df['category'] = df['exam_score'].apply(get_cat)
        counts = df['category'].value_counts()

        categories = ['Excellent', 'Good', 'Average', 'Poor']
        category_counts = [int(counts.get(cat, 0)) for cat in categories]

        return jsonify({'labels': categories, 'counts': category_counts})
    except Exception as e:
        logger.error(f"Error in category distribution: {e}")
        return jsonify({'error': str(e)}), 500


# API endpoint for habits impact chart
@app.route('/api/habits-impact')
def api_habits_impact():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "student_data_cleaned.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Dataset not found'}), 404

        df = pd.read_csv(csv_path)

        # Define groups for comparison
        groups = ['Study Hours', 'Sleep Hours', 'Social Media', 'Exercise']

        # Study: Low (<3hrs) vs High (>=5hrs)
        low_study = df[df['study_hours_per_day'] < 3]['exam_score'].mean()
        high_study = df[df['study_hours_per_day'] >= 5]['exam_score'].mean()

        # Sleep: Poor (<6hrs) vs Good (7-9hrs)
        low_sleep = df[df['sleep_hours'] < 6]['exam_score'].mean()
        high_sleep = df[(df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)]['exam_score'].mean()

        # Social Media: Low (<2hrs) vs High (>4hrs)
        low_social = df[df['social_media_hours'] <= 2]['exam_score'].mean()
        high_social = df[df['social_media_hours'] > 4]['exam_score'].mean()

        # Exercise: Low (<2 days) vs High (>=4 days)
        low_exercise = df[df['exercise_frequency'] < 2]['exam_score'].mean()
        high_exercise = df[df['exercise_frequency'] >= 4]['exam_score'].mean()

        low_scores = [round(low_study, 1), round(low_sleep, 1), round(low_social, 1), round(low_exercise, 1)]
        high_scores = [round(high_study, 1), round(high_sleep, 1), round(high_social, 1), round(high_exercise, 1)]

        return jsonify({
            'groups': groups,
            'low_scores': low_scores,
            'high_scores': high_scores
        })
    except Exception as e:
        logger.error(f"Error in habits impact: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    app.run(debug=debug_mode)
