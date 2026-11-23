import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE = os.path.dirname(os.path.abspath(__file__))

# app.py uses the same order
FEATURES = [
    'age', 'study_hours_per_day', 'social_media_hours',
    'netflix_hours', 'sleep_hours', 'exercise_frequency',
    'mental_health_rating'
]
TARGET = 'exam_score'

NAMES = {
    'age': 'Age',
    'study_hours_per_day': 'Study Hours/Day',
    'social_media_hours': 'Social Media Hours',
    'netflix_hours': 'Netflix Hours',
    'sleep_hours': 'Sleep Hours',
    'exercise_frequency': 'Exercise Frequency',
    'mental_health_rating': 'Mental Health Rating',
}

data = pd.read_csv(os.path.join(BASE, 'data', 'student_habits_performance.csv'))
print("shape:", data.shape)

# only the columns the form asks for
data = data[FEATURES + [TARGET]]
print("missing values:", data.isnull().sum().sum())

# dashboard reads this file
data.to_csv(os.path.join(BASE, 'data', 'student_data_cleaned.csv'), index=False)

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(f"train: {len(X_train)}, test: {len(X_test)}")

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []
pipes = {}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2').mean()

    results.append({'model': name, 'r2': r2, 'mae': mae, 'rmse': rmse, 'cv_r2': cv_r2})
    pipes[name] = pipe
    print(f"{name:18s}  R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  CV R2={cv_r2:.4f}")

# pick by cv score, not test score
results.sort(key=lambda r: r['cv_r2'], reverse=True)
best = results[0]
best_pipe = pipes[best['model']]
print("\nbest model:", best['model'])

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
joblib.dump(best_pipe, os.path.join(BASE, 'models', 'student_model.pkl'))

# inputs are scaled so coef = marks change for +1 std of that habit
model = best_pipe.named_steps['model']
if hasattr(model, 'coef_'):
    effects = model.coef_
else:
    effects = model.feature_importances_

importance = {
    NAMES[f]: round(float(v), 2)
    for f, v in sorted(zip(FEATURES, effects), key=lambda t: abs(t[1]), reverse=True)
}

info = {
    'type': best['model'],
    'r2_score': round(best['r2'], 4),
    'mae': round(best['mae'], 2),
    'rmse': round(best['rmse'], 2),
    'cv_r2': round(best['cv_r2'], 4),
    'features': list(NAMES.values()),
    'feature_importance': importance,
    'comparison': [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        for r in results
    ],
}

with open(os.path.join(BASE, 'models', 'model_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print("saved models/student_model.pkl and models/model_info.json")
