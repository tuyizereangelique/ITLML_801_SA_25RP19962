"""
predict.py
Virtual prediction script to compare with frontend results
Matches frontend HTML inputs and Flask app preprocessing exactly
"""

import joblib
import json
import pandas as pd
import numpy as np


MODEL_PATH = 'deployment/best_model.pkl'
CLASS_NAMES_PATH = 'deployment/class_names.json'

# Load model
model = joblib.load(MODEL_PATH)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_labels_list = json.load(f)

REQUIRED_COLUMNS = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
    'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'
]

NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

ENCODERS = {
    'sex': {'Male': 1, 'Female': 0},
    'cp': {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-Anginal Pain': 2,
        'Asymptomatic': 3
    },
    'restecg': {'Normal': 0, 'ST-T abnormality': 1, 'LV hypertrophy': 2},
    'exang': {'No': 0, 'Yes': 1},
    'slope': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
    'thal': {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
}


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[REQUIRED_COLUMNS]

    # Numeric conversion
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # fbs handling (frontend sends TRUE/FALSE or 0/1)
    df['fbs'] = df['fbs'].apply(
        lambda x: 1 if str(x).lower() in ['1', 'true', 'yes'] else 0
    )

    # Encode categorical values
    for col, mapping in ENCODERS.items():
        df[col] = df[col].astype(str).str.strip().map(mapping)

    return df


# ---------------- Three sample inputs ----------------
sample_inputs = [
    {
        'age': 50,
        'sex': 'Male',
        'cp': 'Typical Angina',
        'trestbps': 150,
        'chol': 250,
        'fbs': 'FALSE',
        'restecg': 'ST-T abnormality',
        'thalach': 150,
        'exang': 'No',
        'oldpeak': 1,
        'slope': 'Flat',
        'ca': 0,
        'thal': 'Normal'
    },
    {
        'age': 60,
        'sex': 'Female',
        'cp': 'Non-Anginal Pain',
        'trestbps': 140,
        'chol': 280,
        'fbs': 'TRUE',
        'restecg': 'Normal',
        'thalach': 130,
        'exang': 'Yes',
        'oldpeak': 2.5,
        'slope': 'Downsloping',
        'ca': 1,
        'thal': 'Fixed Defect'
    },
    {
        'age': 45,
        'sex': 'Male',
        'cp': 'Asymptomatic',
        'trestbps': 120,
        'chol': 200,
        'fbs': 'FALSE',
        'restecg': 'LV hypertrophy',
        'thalach': 160,
        'exang': 'No',
        'oldpeak': 0.5,
        'slope': 'Upsloping',
        'ca': 0,
        'thal': 'Reversible Defect'
    }
]

# Preprocess all samples
df = pd.DataFrame(sample_inputs)
df = preprocess_input(df)

print("INPUT TO MODEL:")
print(df)

preds = model.predict(df)
pred_probas = model.predict_proba(df)
classes = list(model.classes_)

for idx, (pred, pred_proba) in enumerate(zip(preds, pred_probas), start=1):
    print(f"\n--- Prediction for Sample {idx} ---")
    print("Predicted class:", pred)
    print("Probabilities:")
    for cls, p in zip(classes, pred_proba):
        print(f"{cls}: {round(p * 100, 1)}%")
