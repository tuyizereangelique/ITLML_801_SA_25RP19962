Heart-Disease-Risk-Prediction-System

Machine Learning Project to predict Heart Disease Risk

Description

This project is a machine learning–based system designed to predict the risk of heart disease in patients using medical data such as age, sex, blood pressure, cholesterol, chest pain type, and more. It helps healthcare professionals and users assess potential heart disease risk early and make informed decisions.

Features

Predicts heart disease risk using a trained ML model

REST API endpoint (/api/predict) for integration

User-friendly web interface for data input

Displays predicted class and probability for each outcome

Model loading using saved .pkl files

Technologies Used

Python

Flask

Machine Learning

NumPy

Pandas

Scikit-learn

How to Use

Clone or download the repository

Create a Python virtual environment and install required dependencies (pip install -r requirements.txt)

Run the Flask application or predict.py script

Enter patient data through the web interface or API

View the predicted heart disease risk result

If you want, I can also add a small section with example input and output so users immediately know what to expect — it makes the README even more practical. Do you want me to do that?

yes

Perfect! Here’s your updated README with an Example Input/Output section added, keeping the same style as before:

Heart-Disease-Risk-Prediction-System

Machine Learning Project to predict Heart Disease Risk

Description

This project is a machine learning–based system designed to predict the risk of heart disease in patients using medical data such as age, sex, blood pressure, cholesterol, chest pain type, and more. It helps healthcare professionals and users assess potential heart disease risk early and make informed decisions.

Features

Predicts heart disease risk using a trained ML model

REST API endpoint (/api/predict) for integration

User-friendly web interface for data input

Displays predicted class and probability for each outcome

Model loading using saved .pkl files

Technologies Used

Python

Flask

Machine Learning

NumPy

Pandas

Scikit-learn

How to Use

Clone or download the repository

Create a Python virtual environment and install required dependencies:

python -m venv env
source env/bin/activate   # macOS/Linux
env\Scripts\activate      # Windows
pip install -r requirements.txt


Run the Flask application or predict.py script

Enter patient data through the web interface or API

View the predicted heart disease risk result

Example Input

Sample data for prediction:

{
  "age": 55,
  "sex": "Male",
  "cp": "Typical Angina",
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": "Normal",
  "thalach": 150,
  "exang": "No",
  "oldpeak": 1.2,
  "slope": "Flat",
  "ca": 0,
  "thal": "Normal"
}

Example Output
Predicted class: no disease
Probabilities:
no disease: 85.0%
disease: 15.0%
