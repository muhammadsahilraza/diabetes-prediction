# app.py — Streamlit web app for Diabetes Prediction
# Loads the trained XGBoost model + scaler and serves real-time predictions
# Developed by Muhammad Sahil Raza

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetes Prediction | Muhammad Sahil Raza",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Load Model & Scaler ----------------
@st.cache_resource  # Cache so model loads only once
def load_artifacts():
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        st.error("❌ model.pkl or scaler.pkl not found. Please run the notebook first.")
        st.stop()
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem; font-weight: 700; color: #0E4C92;
            text-align: center; margin-bottom: 5px;
        }
        .subtitle {
            text-align: center; color: #555; margin-bottom: 15px; font-size: 1rem;
        }
        .author-line {
            text-align: center; color: #444; font-size: 1rem;
            margin-bottom: 25px;
        }
        .author-name {
            font-size: 1.4rem;
            font-weight: 900;
            color: #0E4C92;
            letter-spacing: 0.6px;
            text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #0E4C92; color: white;
            border-radius: 8px; padding: 10px 24px;
            font-weight: 600; width: 100%;
        }
        .stButton>button:hover { background-color: #08386e; color: white; }
        .result-box {
            padding: 20px; border-radius: 10px; text-align: center;
            font-size: 1.2rem; font-weight: 600; margin-top: 20px;
        }
        .footer-text {
            text-align: center; color: #666; font-size: 0.9rem;
            margin-top: 5px;
        }
        .footer-name {
            font-weight: 800; color: #0E4C92;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="main-title">🩸 Diabetes Prediction System 💉</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details below to estimate diabetes risk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="author-line">👨‍⚕️ Developed by '
    '<span class="author-name">Muhammad Sahil Raza</span></div>',
    unsafe_allow_html=True
)

# ---------------- Input Form ----------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose     = st.number_input("Glucose (mg/dL)", min_value=0, max_value=250, value=120, step=1)
        bp          = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, step=1)
        skin        = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
        bmi     = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age     = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

    submit = st.form_submit_button("🔍 Predict")

# ---------------- Prediction Logic ----------------
if submit:
    # Arrange features in EXACT order the model was trained on
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Scale using the SAME scaler used during training
    features_scaled = scaler.transform(features)

    # Predict
    prediction  = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.markdown(
            f'<div class="result-box" style="background-color:#FDE2E2; color:#B00020;">'
            f'⚠️ High Risk of Diabetes<br>Probability: {probability:.2%}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.warning("Please consult a doctor for proper diagnosis.")
    else:
        st.markdown(
            f'<div class="result-box" style="background-color:#E6F4EA; color:#137333;">'
            f'✅ Low Risk of Diabetes<br>Probability: {probability:.2%}'
            f'</div>',
            unsafe_allow_html=True
        )
        st.success("Keep up the healthy lifestyle!")

    # Show input summary
    with st.expander("📋 View Input Summary"):
        input_df = pd.DataFrame({
            "Feature": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
            "Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        })
        st.dataframe(input_df, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    '<div class="footer-text">Built with ❤️ by '
    '<span class="footer-name">Muhammad Sahil Raza</span></div>',
    unsafe_allow_html=True
)
st.caption("⚠️ This tool is for educational purposes only and not a substitute for professional medical advice.")