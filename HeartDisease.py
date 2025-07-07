import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")


try:
    model = joblib.load('heart_disease_model_3.joblib')
    scaler = joblib.load('heart_disease_scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'heart_disease_model.pkl' and 'scaler.pkl' are in the same directory as this app.")
    st.stop()

# Define metric features
metric_features = [
    "BMI", "PhysicalHealth", "MentalHealth", "SleepTime",
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
    "Diabetic", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"
]

# Streamlit app
st.title("Heart Disease Prediction App")
st.write("Enter the following details to predict the likelihood of heart disease.")

# Create input fields
st.subheader("Input Features")

# Numerical features
bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, help="Body Mass Index (e.g., 18.5–24.9 is normal)")
physical_health = st.slider("Physical Health (days unwell in past 30 days)", min_value=0, max_value=30, value=0, step=1)
mental_health = st.slider("Mental Health (days unwell in past 30 days)", min_value=0, max_value=30, value=0, step=1)
sleep_time = st.slider("Sleep Time (hours per night)", min_value=1, max_value=24, value=7, step=1)

# Binary features (Yes/No)
smoking = st.selectbox("Smoking", options=["No", "Yes"], help="Have you smoked at least 100 cigarettes in your life?")
alcohol_drinking = st.selectbox("Alcohol Drinking", options=["No", "Yes"], help="Heavy drinking (e.g., 14+ drinks/week for men, 7+ for women)")
stroke = st.selectbox("Stroke", options=["No", "Yes"], help="Have you ever had a stroke?")
diff_walking = st.selectbox("Difficulty Walking", options=["No", "Yes"], help="Do you have serious difficulty walking or climbing stairs?")
physical_activity = st.selectbox("Physical Activity", options=["No", "Yes"], help="Physical activity in past 30 days (not including job)")
asthma = st.selectbox("Asthma", options=["No", "Yes"], help="Do you have asthma?")
kidney_disease = st.selectbox("Kidney Disease", options=["No", "Yes"], help="Do you have kidney disease (not including stones or infections)?")
skin_cancer = st.selectbox("Skin Cancer", options=["No", "Yes"], help="Do you have a history of skin cancer?")

# Diabetic feature (with additional options)
diabetic = st.selectbox("Diabetic", options=["No", "Yes", "No, Borderline Diabetes", "Yes (During Pregnancy)"], help="Diabetes status")

# Convert inputs to model-compatible format
input_data = {
    "BMI": bmi,
    "PhysicalHealth": physical_health,
    "MentalHealth": mental_health,
    "SleepTime": sleep_time,
    "Smoking": 1 if smoking == "Yes" else 0,
    "AlcoholDrinking": 1 if alcohol_drinking == "Yes" else 0,
    "Stroke": 1 if stroke == "Yes" else 0,
    "DiffWalking": 1 if diff_walking == "Yes" else 0,
    "Diabetic": 1 if diabetic == "Yes" else 0.5 if diabetic in ["No, Borderline Diabetes", "Yes (During Pregnancy)"] else 0,
    "PhysicalActivity": 1 if physical_activity == "Yes" else 0,
    "Asthma": 1 if asthma == "Yes" else 0,
    "KidneyDisease": 1 if kidney_disease == "Yes" else 0,
    "SkinCancer": 1 if skin_cancer == "Yes" else 0
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=metric_features)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    # Get probability for class 1
    proba = model.predict_proba(input_scaled)[:, 1][0]

    best_threshold = 0.35
    prediction = 1 if proba >= best_threshold else 0
    
    # Display result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Heart Disease: Yes")
        st.warning("Please consult a healthcare professional for a thorough evaluation.")
    else:
        st.success("Heart Disease: No ")
        st.info("This is a prediction based on the model. Regular check-ups are recommended.")

    # Display feature importance (optional)
    st.subheader("Model Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': metric_features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(feature_importance)

# Instructions
st.markdown("""
### Instructions
1. Adjust the sliders and select options to input your data.
2. Click the **Predict** button to see the result.
3. The model uses a RandomForestClassifier trained on the 2020 heart disease dataset.
4. Ensure the model and scaler files (`heart_disease_model.pkl`, `scaler.pkl`) are in the same directory as this app.
""")