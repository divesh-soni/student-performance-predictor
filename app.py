from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

BASE = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE, "data", "student_data_cleaned.csv")
DB = os.path.join(BASE, 'predictions.db')

# model_info.json is made by train_model.py
try:
    with open(os.path.join(BASE, "models", "model_info.json")) as f:
        MODEL_INFO = json.load(f)
except FileNotFoundError:
    print("model_info.json not found, run train_model.py first")
    MODEL_INFO = {'type': 'Not trained', 'r2_score': 0, 'mae': 0, 'rmse': 0,
                  'features': [], 'feature_importance': {}}

MODEL_PATH = os.path.join(BASE, "models", "student_model.pkl")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("model loaded")
except FileNotFoundError:
    print("model not found, run train_model.py first")
    pipeline = None
except Exception as e:
    print("could not load model:", e)
    pipeline = None

# same order as train_model.py
FEATURE_NAMES = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                 'sleep_hours', 'exercise_frequency', 'mental_health_rating']


def init_db():
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions
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


def get_features(data):
    features = {
        'age': int(data['age']),
        'study_hours_per_day': float(data['study_hours']),
        'social_media_hours': float(data['social_hours']),
        'netflix_hours': float(data['netflix_hours']),
        'sleep_hours': float(data['sleep_hours']),
        'exercise_frequency': int(data['exercise_freq']),
        'mental_health_rating': int(data['mental_health'])
    }
    return features


def save_prediction(data, score, category):
    try:
        f = get_features(data)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(DB)
        conn.execute('''INSERT INTO predictions
                     (student_name, age, study_hours, social_media_hours, netflix_hours,
                      sleep_hours, exercise_frequency, mental_health_rating,
                      predicted_score, performance_category, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (data['student_name'], f['age'], f['study_hours_per_day'],
                      f['social_media_hours'], f['netflix_hours'], f['sleep_hours'],
                      f['exercise_frequency'], f['mental_health_rating'],
                      score, category, now))
        conn.commit()
        conn.close()
        log.info("saved prediction for " + data['student_name'])
    except Exception as e:
        log.error("could not save prediction: " + str(e))


def get_all_predictions():
    try:
        conn = sqlite3.connect(DB)
        rows = conn.execute('SELECT * FROM predictions ORDER BY timestamp DESC').fetchall()
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
    except Exception as e:
        log.error("could not read predictions: " + str(e))
        return []


def delete_prediction(pred_id):
    try:
        conn = sqlite3.connect(DB)
        conn.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log.error("could not delete: " + str(e))
        return False


def load_data():
    if not os.path.exists(DATA_CSV):
        return None
    return pd.read_csv(DATA_CSV)


def get_dataset_stats():
    try:
        df = load_data()
        if df is None:
            log.error("dataset not found")
            return None

        total = len(df)
        stats = {
            'total_students': total,
            'avg_score': round(df['exam_score'].mean(), 1),
            'highest_score': df['exam_score'].max(),
            'lowest_score': df['exam_score'].min(),
            'avg_study': round(df['study_hours_per_day'].mean(), 1),
            'avg_sleep': round(df['sleep_hours'].mean(), 1),
            'avg_social': round(df['social_media_hours'].mean(), 1),
            'avg_netflix': round(df['netflix_hours'].mean(), 1),
            'avg_exercise': round(df['exercise_frequency'].mean(), 1),
            'avg_mental': round(df['mental_health_rating'].mean(), 1),
            'pass_rate': round((df['exam_score'] >= 50).sum() / total * 100, 1),
            'excellent_rate': round((df['exam_score'] >= 85).sum() / total * 100, 1),
        }
        return stats
    except Exception as e:
        log.error("stats error: " + str(e))
        return None


def get_average_stats():
    stats = get_dataset_stats()
    if stats is None:
        # values from the real dataset in case the csv is missing
        return {'avg_score': 69.6, 'avg_study': 3.6, 'avg_sleep': 6.5, 'avg_social': 2.5,
                'avg_exercise': 3.0, 'avg_mental': 5.4, 'avg_netflix': 1.8}

    return {
        'avg_score': stats['avg_score'],
        'avg_study': stats['avg_study'],
        'avg_sleep': stats['avg_sleep'],
        'avg_social': stats['avg_social'],
        'avg_exercise': stats['avg_exercise'],
        'avg_mental': stats['avg_mental'],
        'avg_netflix': stats['avg_netflix'],
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
        return 'F', 'Needs a lot of improvement'


def check_number(errors, value, label, low, high, is_int=False):
    try:
        if is_int:
            num = int(value)
        else:
            num = float(value)
        if num < low or num > high:
            errors.append(label + " must be between " + str(low) + " and " + str(high))
    except (ValueError, TypeError):
        errors.append("Enter a valid value for " + label.lower())


def validate_input(data):
    errors = []

    name = data.get('student_name', '').strip()
    if name == '':
        errors.append("Name is required")
    elif len(name) < 2:
        errors.append("Name must be at least 2 characters")

    check_number(errors, data['age'], "Age", 15, 100, is_int=True)
    check_number(errors, data['study_hours'], "Study hours", 0, 24)
    check_number(errors, data['social_hours'], "Social media hours", 0, 24)
    check_number(errors, data['netflix_hours'], "Entertainment hours", 0, 24)
    check_number(errors, data['sleep_hours'], "Sleep hours", 3, 14)
    check_number(errors, data['exercise_freq'], "Exercise days", 0, 7, is_int=True)
    check_number(errors, data['mental_health'], "Mental health rating", 1, 10, is_int=True)

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


def generate_strengths_weaknesses(f):
    strengths = []
    weaknesses = []

    if f['study_hours_per_day'] >= 5:
        strengths.append("Good study habits, you study 5+ hours a day")
    elif f['study_hours_per_day'] >= 4:
        strengths.append("Decent study routine")
    elif f['study_hours_per_day'] < 3:
        weaknesses.append("Low study hours, try 4-5 hours a day")

    if f['social_media_hours'] <= 2:
        strengths.append("Low social media use")
    elif f['social_media_hours'] <= 3:
        strengths.append("Social media use is okay")
    elif f['social_media_hours'] > 4:
        weaknesses.append("Too much social media, it can hurt focus")

    if 7 <= f['sleep_hours'] <= 9:
        strengths.append("Good sleep (7-9 hours)")
    elif f['sleep_hours'] < 6:
        weaknesses.append("Not enough sleep, can affect concentration")
    elif f['sleep_hours'] > 9:
        weaknesses.append("Sleeping more than 9 hours")

    if f['exercise_frequency'] >= 3:
        strengths.append("Regular exercise")
    elif f['exercise_frequency'] >= 1:
        strengths.append("Some exercise, try to make it 3+ days a week")
    else:
        weaknesses.append("No exercise")

    if f['mental_health_rating'] >= 8:
        strengths.append("Very good mental health")
    elif f['mental_health_rating'] >= 6:
        strengths.append("Good mental health")
    elif f['mental_health_rating'] < 5:
        weaknesses.append("Low mental health rating, maybe talk to a counselor")

    if len(strengths) == 0:
        strengths.append("Better daily habits can improve your score")
    if len(weaknesses) == 0:
        weaknesses.append("Nothing major, keep it up")

    return strengths, weaknesses


def make_rec(area, suggestion, tip):
    return {'area': area, 'suggestion': suggestion, 'actionable_tip': tip}


def generate_recommendations(f, score):
    recs = []

    if f['study_hours_per_day'] < 4:
        recs.append(make_rec('Study Habits',
                             'Increase study time from ' + str(f['study_hours_per_day']) + ' to 4-5 hours a day',
                             'Try pomodoro: 25 min study, 5 min break'))
    elif f['study_hours_per_day'] > 7:
        recs.append(make_rec('Study-Life Balance',
                             'You study ' + str(f['study_hours_per_day']) + ' hours a day, focus on quality',
                             'Take breaks and revise what you studied'))

    if f['social_media_hours'] > 3:
        recs.append(make_rec('Social Media',
                             'Cut social media from ' + str(f['social_media_hours']) + ' to under 2 hours a day',
                             'Use app timers on your phone'))

    if f['sleep_hours'] < 7:
        recs.append(make_rec('Sleep', 'Sleep 7-8 hours',
                             'Sleep at the same time daily and keep the phone away before bed'))
    elif f['sleep_hours'] > 9:
        recs.append(make_rec('Sleep', 'Try to bring sleep down to 7-8 hours',
                             'Wake up at the same time every day'))

    if f['exercise_frequency'] < 2:
        recs.append(make_rec('Exercise', 'Exercise 2-3 times a week', 'Even a 15-20 min walk helps'))

    if f['mental_health_rating'] < 5:
        recs.append(make_rec('Mental Health', 'Take care of your mental health',
                             'Talk to a counselor, friends or family'))

    if score < 50:
        recs.append(make_rec('Academic Help', 'Get some extra help with studies',
                             'Join a study group or ask teachers for help'))

    if len(recs) == 0:
        recs.append(make_rec('Keep Going', 'You are doing well, keep the same habits',
                             'Check your progress every week'))

    return recs


def calculate_percentile(score):
    try:
        df = load_data()
        if df is None:
            return None
        return round((df['exam_score'] < score).sum() / len(df) * 100, 1)
    except Exception as e:
        log.error("percentile error: " + str(e))
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', errors=[], form_data={})

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
    if len(errors) == 0 and pipeline is None:
        errors = ["Model not loaded, run train_model.py first"]
    if len(errors) > 0:
        return render_template('predict.html', errors=errors, form_data=student_data)

    features = get_features(student_data)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    try:
        score = pipeline.predict(X)[0]
        # linear regression can give less than 0 or more than 100
        score = round(float(np.clip(score, 0, 100)), 2)
    except Exception as e:
        log.error("prediction error: " + str(e))
        errors = ["Prediction failed, please try again"]
        return render_template('predict.html', errors=errors, form_data=student_data)

    category = get_performance_category(score)
    strengths, weaknesses = generate_strengths_weaknesses(features)
    recommendations = generate_recommendations(features, score)

    save_prediction(student_data, score, category)

    session['last_prediction'] = {
        'student_name': student_data['student_name'],
        'score': score,
        'category': category,
        'features': features,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations
    }
    return redirect(url_for('report'))


@app.route('/report')
def report():
    prediction = session.get('last_prediction')
    if not prediction:
        return redirect(url_for('predict'))

    percentile = calculate_percentile(prediction['score'])
    if percentile is None:
        percentile = 50
    letter_grade, grade_msg = get_letter_grade(prediction['score'])
    avg_stats = get_average_stats()

    return render_template('report.html', prediction=prediction, avg_stats=avg_stats,
                           percentile=percentile, letter_grade=letter_grade, grade_msg=grade_msg)


@app.route('/history')
def history():
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)


@app.route('/dashboard')
def dashboard():
    stats = get_dataset_stats()
    if stats is None:
        stats = {
            'total_students': 1000, 'avg_score': 69.6, 'highest_score': 100,
            'lowest_score': 18.4, 'avg_study': 3.6, 'avg_sleep': 6.5,
            'avg_social': 2.5, 'avg_netflix': 1.8, 'avg_exercise': 3.0,
            'avg_mental': 5.4, 'pass_rate': 75, 'excellent_rate': 25, 'error': True
        }
    return render_template('dashboard.html', stats=stats, model_info=MODEL_INFO)


@app.route('/about')
def about():
    return render_template('about.html', model_info=MODEL_INFO)


@app.route('/api/delete/<int:pred_id>', methods=['DELETE'])
def api_delete(pred_id):
    success = delete_prediction(pred_id)
    return jsonify({'success': success})


@app.route('/api/score-distribution')
def api_score_distribution():
    try:
        df = load_data()
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404

        bins = [0, 20, 40, 50, 60, 70, 80, 90, 100]
        labels = ['0-20', '21-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
        df['score_range'] = pd.cut(df['exam_score'], bins=bins, labels=labels, right=True)
        counts = df['score_range'].value_counts().reindex(labels).fillna(0).astype(int).tolist()

        return jsonify({'bins': labels, 'counts': counts})
    except Exception as e:
        log.error("score distribution error: " + str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/category-distribution')
def api_category_distribution():
    try:
        df = load_data()
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404

        df['category'] = df['exam_score'].apply(get_performance_category)
        counts = df['category'].value_counts()

        labels = ['Excellent', 'Good', 'Average', 'Poor']
        values = []
        for c in labels:
            values.append(int(counts.get(c, 0)))

        return jsonify({'labels': labels, 'counts': values})
    except Exception as e:
        log.error("category distribution error: " + str(e))
        return jsonify({'error': str(e)}), 500


def avg_score(df, mask):
    return round(df[mask]['exam_score'].mean(), 1)


@app.route('/api/habits-impact')
def api_habits_impact():
    try:
        df = load_data()
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404

        # low group vs high group for each habit
        # for social media the "low" group is the one using it less
        low = [
            avg_score(df, df['study_hours_per_day'] < 3),
            avg_score(df, df['sleep_hours'] < 6),
            avg_score(df, df['social_media_hours'] <= 2),
            avg_score(df, df['exercise_frequency'] < 2),
        ]
        high = [
            avg_score(df, df['study_hours_per_day'] >= 5),
            avg_score(df, (df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)),
            avg_score(df, df['social_media_hours'] > 4),
            avg_score(df, df['exercise_frequency'] >= 4),
        ]

        return jsonify({
            'groups': ['Study Hours', 'Sleep Hours', 'Social Media', 'Exercise'],
            'low_scores': low,
            'high_scores': high
        })
    except Exception as e:
        log.error("habits impact error: " + str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_ENV') != 'production')
