❤️ Heart Disease Prediction using Machine Learning

A complete end-to-end Machine Learning project that predicts the likelihood of heart disease using patient clinical data. The system includes data preprocessing, model training, evaluation, and a simple web interface for real-time prediction.

📌 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early prediction can greatly improve diagnosis, treatment, and patient survival.

This project uses several machine learning algorithms to analyze medical features such as age, chest pain type, cholesterol level, maximum heart rate, and more — to predict whether a patient is at risk of heart disease.

🧠 Machine Learning Pipeline
✔ Data Preprocessing

Handling missing values

Feature encoding (One-Hot Encoding)

Normalization / Standardization

Train–test split

✔ Model Training

Models used:

Logistic Regression

Random Forest Classifier

Optional: SVM, KNN, XGBoost

✔ Model Evaluation

Accuracy

Precision & Recall

F1 Score

ROC AUC Score

Confusion Matrix

The best-performing model is saved as model.pkl / model.joblib.

🌐 Web Application

A simple and user-friendly web interface (Flask/Streamlit) allows users to input values such as:

Age

Sex

Chest Pain Type

Blood Pressure

Cholesterol

Resting ECG

Max Heart Rate

Exercise-induced Angina

Oldpeak

Slope

Number of Major Vessels

Thal

Based on the inputs, the app displays:

✔ Predicted Output (Heart Disease: Yes / No)

✔ Probability Score

✔ Risk Level (Low / Medium / High)

🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Flask / Streamlit

HTML, CSS, JavaScript

Joblib / Pickle

📁 Project Structure
Heart-Disease-Prediction/
│
├── model_training.py
├── model.pkl / model.joblib
├── app.py
├── index.html
├── styles.css
├── script.js
├── requirements.txt
└── README.md

🚀 How to Run the Project
1. Install requirements
pip install -r requirements.txt

2. Train the Model
python model_training.py

3. Run the Web App

Flask:

python app.py


Streamlit:

streamlit run app.py

📊 Results

Successfully trained ML models on heart disease dataset

Achieved strong accuracy and balanced performance

Provides real-time heart disease prediction

User-friendly interface for easy interaction

📎 Future Improvements

Add SHAP explainability for feature insights

Improve UI/UX with animations and charts

Deploy to cloud (Streamlit Cloud, Render, or Netlify + Flask API)

Add multiple model selection options

🤝 Contributions

Pull requests are welcome!
Feel free to open an issue for new features or improvements.

⭐ If you like this project, give it a star on GitHub!
