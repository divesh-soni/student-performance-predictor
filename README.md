# Student Performance Prediction System

A Flask-based web application that predicts student academic performance using machine learning and provides personalized recommendations based on habits and lifestyle factors.

## Features

- **Performance Prediction**: ML-powered predictions based on study habits, sleep, exercise, and mental health
- **Interactive Dashboard**: Visualize score distributions and habit impact analysis
- **Detailed Reports**: Get personalized recommendations and performance insights
- **Prediction History**: Track previous predictions with database storage
- **Comparative Analysis**: See how your performance compares to average statistics

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Student Performance Prediction System"
```

2. **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare the model**
   - Run `preprocessing_02.ipynb` to train and save the model to `models/student_model.pkl`
   - Ensure `data/student_data_cleaned.csv` exists in the data folder

## Usage

1. **Start the application**
```bash
python app.py
```

2. **Open in browser**
```
http://localhost:5000
```

3. **Navigate the app**
   - **Home**: Overview and introduction
   - **Predict**: Input student data to get performance predictions
   - **Dashboard**: View statistical insights and trends
   - **History**: Browse all previous predictions
   - **About**: Learn more about the system

## Project Structure

```
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── data/                       # Data files
│   ├── student_data_cleaned.csv
│   └── student_habits_performance.csv
├── models/                     # Machine learning model
│   └── student_model.pkl       # Trained model (generated)
├── static/                     # Frontend assets
│   ├── style.css
│   └── script.js
├── templates/                  # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── report.html
│   ├── dashboard.html
│   ├── history.html
│   └── about.html
├── EDA 01.ipynb               # Exploratory Data Analysis
└── preprocessing_02.ipynb     # Data preprocessing & model training
```

## Environment Variables

For production deployment, set the following environment variables:

```bash
# Secret key for Flask sessions (generate a secure key)
export FLASK_SECRET_KEY="your-secret-key-here"

# Flask environment
export FLASK_ENV=production
```

Update `app.py` to use environment variables:
```python
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key')
debug = os.getenv('FLASK_ENV') != 'production'
```

## Model Training

To retrain the model:

1. Update data files in the `data/` folder
2. Run `preprocessing_02.ipynb` to preprocess data and train the model
3. The model will be saved as `models/student_model.pkl`

## Features Explained

The prediction model uses these features:
- **Age**: Student's age (15-100 years)
- **Study Hours**: Daily study hours (0-24)
- **Social Media Hours**: Daily social media usage (0-24)
- **Netflix Hours**: Daily streaming hours (0-24)
- **Sleep Hours**: Daily sleep duration (3-14 hours)
- **Exercise Frequency**: Weekly exercise days (0-7)
- **Mental Health Rating**: Self-rated mental health (1-10)

## Performance Categories

- **Excellent**: Score ≥ 85
- **Good**: Score 70-84
- **Average**: Score 50-69
- **Poor**: Score < 50

## Database

The application uses SQLite to store prediction history:
- **File**: `predictions.db` (auto-generated)
- **Table**: `predictions` - Stores student prediction records

## Troubleshooting

### Model not loading
- Ensure `models/student_model.pkl` exists
- Run `preprocessing_02.ipynb` to generate the model

### Data files not found
- Verify `data/student_data_cleaned.csv` exists
- Check file paths in `app.py`

### Port already in use
```bash
# Change port in app.py or use:
python app.py --port 5001
```

## Future Enhancements

- [ ] User authentication system
- [ ] More detailed analytics
- [ ] Export reports as PDF
- [ ] Multi-language support
- [ ] Mobile app version

## License

This project is open source and available under the MIT License.

## Contributors

- Created as an educational project for student performance prediction

## Support

For issues or questions, please open an issue in the repository.
