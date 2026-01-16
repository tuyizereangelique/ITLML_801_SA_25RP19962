from flask import Flask, request, jsonify, render_template
import joblib, json, pandas as pd, numpy as np
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = 'deployment/best_model.pkl'
CLASS_NAMES_PATH = 'deployment/class_names.json'

# Load model
model = joblib.load(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_labels_list = json.load(f)

REQUIRED_COLUMNS = ['age','trestbps','chol','thalach','oldpeak','ca',
                    'sex','cp','fbs','restecg','exang','slope','thal']
NUMERIC_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']

ENCODERS = {
    'sex': {'Male':1,'Female':0},
    'cp': {'Typical Angina':0,'Atypical Angina':1,'Non-Anginal Pain':2,'Asymptomatic':3},
    'restecg': {'Normal':0,'ST-T abnormality':1,'LV hypertrophy':2},
    'exang': {'No':0,'Yes':1},
    'slope': {'Upsloping':0,'Flat':1,'Downsloping':2},
    'thal': {'Normal':1,'Fixed Defect':2,'Reversible Defect':3},
    'fbs': {'FALSE':0,'TRUE':1}
}

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_COLUMNS]
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col, mapping in ENCODERS.items():
        df[col] = df[col].astype(str).str.strip().map(mapping)
    return df

def risk_color(prob):
    if prob >= 70: return '#dc3545'  # red
    elif prob >= 50: return '#fd7e14'  # orange
    elif prob >= 30: return '#ffc107'  # yellow
    else: return '#28a745'  # green

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = preprocess_input(df)
    pred = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]
    classes = list(model.classes_)

    probabilities = []
    for cls, p in zip(classes, pred_proba):
        probabilities.append({
            'class_name': str(cls),
            'probability': round(float(p)*100,1),
            'color': risk_color(p*100)
        })

    return jsonify({
        'success': True,
        'prediction': {'class_name': str(pred)},
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)
