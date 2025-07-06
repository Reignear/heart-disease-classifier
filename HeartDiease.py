import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model = joblib.load('heart_disease_model.joblib')
    expected_features = model.feature_names_in_
    st.session_state['expected_features'] = expected_features
except Exception as e:
    st.error(f"Error loading model: {e}")
    expected_features = []

st.title('Heart Disease Prediction')
st.write('Enter the following details to predict the risk of heart disease:')


input_data = {}

st.header('Health Metrics')
col1, col2 = st.columns(2)
with col1:
    input_data['BMI'] = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    input_data['PhysicalHealth'] = st.slider('Physical Health (days where health not good, past 30 days)', 0, 30, 0)
with col2:
    input_data['MentalHealth'] = st.slider('Mental Health (days where health not good, past 30 days)', 0, 30, 0)
    input_data['SleepTime'] = st.slider('Average Sleep Time (hours/day)', 1, 24, 7)

st.header('Demographics')
col1, col2 = st.columns(2)
with col1:
    sex = st.radio('Sex', ['Female', 'Male'])
    input_data['Sex_Female'] = 1 if sex == 'Female' else 0
    input_data['Sex_Male'] = 1 if sex == 'Male' else 0
    
    age_categories = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
                      '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
    age = st.selectbox('Age Category', age_categories)
    for category in age_categories:
        input_data[f'AgeCategory_{category}'] = 1 if age == category else 0

with col2:
    race_categories = ['White', 'Black', 'Asian', 'Hispanic', 'American Indian/Alaskan Native', 'Other']
    race = st.selectbox('Race', race_categories)
    for category in race_categories:
        input_data[f'Race_{category}'] = 1 if race == category else 0

# Lifestyle Factors Section
st.header('Lifestyle Factors')
col1, col2, col3 = st.columns(3)
with col1:
    smoking = st.radio('Smoking', ['No', 'Yes'])
    input_data['Smoking_no'] = 1 if smoking == 'No' else 0
    input_data['Smoking_yes'] = 1 if smoking == 'Yes' else 0
with col2:
    alcohol = st.radio('Alcohol Drinking', ['No', 'Yes'])
    input_data['AlcoholDrinking_No'] = 1 if alcohol == 'No' else 0
    input_data['AlcoholDrinking_Yes'] = 1 if alcohol == 'Yes' else 0
with col3:
    activity = st.radio('Physical Activity', ['No', 'Yes'])
    input_data['PhysicalActivity_No'] = 1 if activity == 'No' else 0
    input_data['PhysicalActivity_Yes'] = 1 if activity == 'Yes' else 0

st.header('Medical Conditions')
col1, col2 = st.columns(2)
with col1:
    stroke = st.radio('Stroke', ['No', 'Yes'])
    input_data['Stroke_No'] = 1 if stroke == 'No' else 0
    input_data['Stroke_Yes'] = 1 if stroke == 'Yes' else 0
    
    diff_walking = st.radio('Difficulty Walking', ['No', 'Yes'])
    input_data['DiffWalking_No'] = 1 if diff_walking == 'No' else 0
    input_data['DiffWalking_Yes'] = 1 if diff_walking == 'Yes' else 0
    
    asthma = st.radio('Asthma', ['No', 'Yes'])
    input_data['Asthma_No'] = 1 if asthma == 'No' else 0
    input_data['Asthma_Yes'] = 1 if asthma == 'Yes' else 0

with col2:
    kidney_disease = st.radio('Kidney Disease', ['No', 'Yes'])
    input_data['KidneyDisease_No'] = 1 if kidney_disease == 'No' else 0
    input_data['KidneyDisease_Yes'] = 1 if kidney_disease == 'Yes' else 0
    
    skin_cancer = st.radio('Skin Cancer', ['No', 'Yes'])
    input_data['SkinCancer_No'] = 1 if skin_cancer == 'No' else 0
    input_data['SkinCancer_Yes'] = 1 if skin_cancer == 'Yes' else 0
    
    diabetic_options = ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)']
    diabetic = st.selectbox('Diabetic', diabetic_options)
    for option in diabetic_options:
        input_data[f'Diabetic_{option}'] = 1 if diabetic == option else 0

st.header('Self-Reported Health')
health_options = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
gen_health = st.select_slider('General Health', options=health_options, value='Good')
for option in health_options:
    input_data[f'GenHealth_{option}'] = 1 if gen_health == option else 0


if st.button('Predict Heart Disease Risk'):

    input_df = pd.DataFrame([input_data])
    
    st.write("Debug information (will be hidden in production):")
    st.write(f"Input features: {input_df.columns.tolist()}")
    
    if 'expected_features' in st.session_state and len(st.session_state['expected_features']) > 0:
        st.write(f"Model expects: {st.session_state['expected_features'].tolist()}")
        
        for feature in st.session_state['expected_features']:
            if feature not in input_df.columns:
                input_df[feature] = 0 
  
        input_df = input_df[st.session_state['expected_features']]
        
        try:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            st.subheader('Prediction Result')
            if prediction[0] == 1:
                st.error('⚠️ High risk of heart disease detected')
                st.write(f'Probability: {prediction_proba[0][1]:.2%}')
            else:
                st.success('✅ Low risk of heart disease')
                st.write(f'Probability: {prediction_proba[0][0]:.2%}')
        
            st.subheader('Explanation')
            st.write("""
            This prediction is based on your input data compared to patterns learned from thousands of patient records.
            Risk factors for heart disease include age, lifestyle factors like smoking and physical inactivity,
            and medical conditions like diabetes and hypertension.
            """)
            st.write("Based on these factors, the model has made its prediction.")
            st.subheader('Recommendations')
            st.write("""
            - Consult with a healthcare provider for a thorough evaluation
            - Maintain a healthy diet and regular exercise routine
            - Monitor blood pressure and cholesterol levels regularly
            - Avoid smoking and limit alcohol consumption
            """)
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded properly or feature names not available")