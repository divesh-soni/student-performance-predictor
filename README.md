# Student Performance Prediction System

Flask web app that predicts a student's exam score (0-100) from daily habits like study hours, sleep, social media and exercise. It also shows a report with grade, percentile, strengths/weaknesses and some tips.

Dataset: student habits and performance dataset from Kaggle (1000 students).

## Pages

- **Predict** - fill the form and get your predicted score
- **Report** - score, grade, comparison with the average student, tips
- **Analytics** - charts from the dataset (Chart.js)
- **History** - all past predictions (saved in SQLite), can delete them
- **About** - model details and coefficients

## How to run

```bash
python -m venv venv
venv\Scripts\activate          # windows
source venv/bin/activate       # mac/linux

pip install -r requirements.txt
python train_model.py
python app.py
```

Then open http://localhost:5000

`train_model.py` creates `models/student_model.pkl`, `models/model_info.json` and `data/student_data_cleaned.csv`.

## Folder structure

```
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── student_habits_performance.csv
│   └── student_data_cleaned.csv
├── models/
│   ├── student_model.pkl
│   └── model_info.json
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── report.html
│   ├── dashboard.html
│   ├── history.html
│   └── about.html
├── EDA 01.ipynb
└── preprocessing_02.ipynb
```

## Model

Features used: age, study hours, social media hours, Netflix hours, sleep hours, exercise days per week, mental health rating (1-10). Other columns like gender, diet and internet quality were dropped because the form doesn't ask for them.

- 80/20 train test split (`random_state=123`)
- `StandardScaler` + model in a sklearn `Pipeline`
- tried 3 models and picked the best 5-fold CV R²

| Model | R² (test) | MAE | RMSE | CV R² |
|---|---|---|---|---|
| Linear Regression | 0.901 | 4.41 | 5.46 | 0.888 |
| Random Forest | 0.858 | 5.33 | 6.55 | 0.852 |
| Decision Tree | 0.770 | 6.46 | 8.33 | 0.730 |

Linear Regression did best. Coefficients (scaled data): study hours +13.8, mental health +5.5, social media -3.0, exercise +2.9, Netflix -2.5, sleep +2.4, age about 0.

## Categories

- Excellent: 85 and above
- Good: 70 - 84
- Average: 50 - 69
- Poor: below 50

## Notes

- Predictions are stored in `predictions.db` (created automatically).
- If the model is missing, run `python train_model.py` again.
- To use another port: `app.run(port=5001)` at the end of `app.py`.
