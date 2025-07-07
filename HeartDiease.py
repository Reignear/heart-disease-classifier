import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('heart_disease_model_2.joblib')

st.title('Heart Disease Prediction')
st.write('Enter the following details to predict the risk of heart disease:')

# Create a dictionary to store all input values
input_data = {}

# Health Metrics Section
st.header('Health Metrics')
col1, col2 = st.columns(2)
with col1:
    input_data['BMI'] = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    input_data['PhysicalHealth'] = st.slider('Physical Health (days where health not good, past 30 days)', 0, 30, 0)
with col2:
    input_data['MentalHealth'] = st.slider('Mental Health (days where health not good, past 30 days)', 0, 30, 0)
    input_data['SleepTime'] = st.slider('Average Sleep Time (hours/day)', 1, 24, 7)

# Demographics Section
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

# Medical Conditions Section
st.header('Medical Conditions')
col1, col2 = st.columns(2)
with col1:
    stroke = st.radio('Stroke', ['No', 'Yes'])
    input_data['Stroke'] = 0 if stroke == 'No' else 1
    
    diff_walking = st.radio('Difficulty Walking', ['No', 'Yes'])
    input_data['DiffWalking'] = 0 if diff_walking == 'No' else 1
    
    asthma = st.radio('Asthma', ['No', 'Yes'])
    input_data['Asthma'] = 0 if asthma == 'No' else 1

with col2:
    kidney_disease = st.radio('Kidney Disease', ['No', 'Yes'])
    input_data['KidneyDisease'] = 0 if kidney_disease == 'No' else 1
    
    skin_cancer = st.radio('Skin Cancer', ['No', 'Yes'])
    input_data['SkinCancer'] = 0 if skin_cancer == 'No' else 1
    
    diabetic_options = ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)']
    diabetic = st.selectbox('Diabetic', diabetic_options)
    if diabetic == 'No':
        input_data['Diabetic'] = 0
    elif diabetic == 'Yes':
        input_data['Diabetic'] = 1
    else:  # Borderline or during pregnancy
        input_data['Diabetic'] = 0.5

# Self-Reported Health Section
st.header('Self-Reported Health')
health_options = ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
gen_health = st.select_slider('General Health', options=health_options, value='Good')
for option in health_options:
    input_data[f'GenHealth_{option}'] = 1 if gen_health == option else 0

# Make prediction
if st.button('Predict Heart Disease Risk'):
    # Convert the input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display result
    
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error('⚠️ High risk of heart disease detected')
        st.write(f'Probability: {prediction_proba[0][1]:.2%}')
    else:
        st.success('✅ Low risk of heart disease')
        st.write(f'Probability: {prediction_proba[0][0]:.2%}')
    
    # Display explanation
    st.subheader('Explanation')
    st.write("""
    This prediction is based on your input data compared to patterns learned from thousands of patient records.
    Risk factors for heart disease include age, lifestyle factors like smoking and physical inactivity,
    and medical conditions like diabetes and hypertension.
    """)
    
    # Recommendations
    st.subheader('Recommendations')
    st.write("""
    - Consult with a healthcare provider for a thorough evaluation
    - Maintain a healthy diet and regular exercise routine
    - Monitor blood pressure and cholesterol levels regularly
    - Avoid smoking and limit alcohol consumption
    """)