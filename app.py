import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Heart Disease Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**About:**")
st.sidebar.markdown("This app predicts the likelihood of heart disease based on medical features.")
st.sidebar.markdown("**Dataset:** UCI Heart Disease Dataset")
st.sidebar.markdown("**Models:** Logistic Regression, Random Forest, SVM")

# Main title
st.title("🫀 Heart Disease Prediction")
st.markdown("Enter your medical information below to get a prediction.")

# Input form
st.header("Medical Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

# Prediction button
if st.button("Predict Heart Disease"):
    try:
        # Load model and scaler
        if not os.path.exists('models/heart_model.pkl') or not os.path.exists('models/scaler.pkl'):
            st.error("Model files not found. Please run the training script first.")
            st.stop()
        
        model = joblib.load('models/heart_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Prepare input
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        
        # Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display result
        if prediction == 1:
            st.markdown(f'<div class="prediction-box positive"><h2>⚠️ High Risk of Heart Disease</h2><p>Probability: {probability:.2%}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-box negative"><h2>✅ Low Risk of Heart Disease</h2><p>Probability: {probability:.2%}</p></div>', unsafe_allow_html=True)
        
        # Risk factor analysis
        st.header("Risk Factor Analysis")
        risk_factors = []
        if age > 60:
            risk_factors.append("Age > 60")
        if chol > 240:
            risk_factors.append("High Cholesterol (>240 mg/dl)")
        if trestbps > 140:
            risk_factors.append("High Blood Pressure (>140 mm Hg)")
        if thalach < 120:
            risk_factors.append("Low Max Heart Rate (<120)")
        if oldpeak > 2:
            risk_factors.append("High ST Depression (>2)")
        if ca > 0:
            risk_factors.append("Major Vessels Affected")
        
        if risk_factors:
            st.write("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No major risk factors identified.")
        
        # Download report
        report = f"""
Heart Disease Prediction Report
===============================

Patient Information:
- Age: {age}
- Sex: {"Male" if sex else "Female"}
- Chest Pain Type: {["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][cp]}
- Resting BP: {trestbps} mm Hg
- Cholesterol: {chol} mg/dl
- Fasting BS >120: {"Yes" if fbs else "No"}
- Resting ECG: {["Normal", "ST-T abnormality", "LVH"][restecg]}
- Max Heart Rate: {thalach}
- Exercise Angina: {"Yes" if exang else "No"}
- ST Depression: {oldpeak}
- Slope: {["Upsloping", "Flat", "Downsloping"][slope]}
- Major Vessels: {ca}
- Thalassemia: {["Normal", "Fixed Defect", "Reversible Defect"][thal]}

Prediction: {"High Risk" if prediction else "Low Risk"}
Probability: {probability:.2%}

Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}
"""
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name="heart_disease_report.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This is a predictive model and not a substitute for professional medical advice.*")
