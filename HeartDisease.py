import streamlit as st
import pandas as pd
import joblib 

# Load the trained model (expecting 13 features)
model = joblib.load('heart_disease_model_2.joblib')

# Define the exact features used during training
expected_features = [
    'BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime',
    'Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'Stroke',
    'DiffWalking', 'Diabetic', 'Asthma', 'KidneyDisease', 'SkinCancer'
]

st.title('Heart Disease Prediction')
st.write('Enter the following details to predict the risk of heart disease:')

# Create a dictionary to store input values
input_data = {}

# Health Metrics Section
st.header('Health Metrics')
col1, col2 = st.columns(2)
with col1:
    input_data['BMI'] = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    input_data['PhysicalHealth'] = st.slider('Physical Health (days not good)', 0, 30, 0)
with col2:
    input_data['MentalHealth'] = st.slider('Mental Health (days not good)', 0, 30, 0)
    input_data['SleepTime'] = st.slider('Sleep Time (hours/day)', 1, 24, 7)

# Lifestyle Section
st.header('Lifestyle Factors')
col1, col2 = st.columns(2)
with col1:
    input_data['Smoking'] = 1 if st.radio('Do you smoke?', ['No', 'Yes']) == 'Yes' else 0
    input_data['AlcoholDrinking'] = 1 if st.radio('Do you drink alcohol?', ['No', 'Yes']) == 'Yes' else 0
with col2:
    input_data['PhysicalActivity'] = 1 if st.radio('Physical activity?', ['No', 'Yes']) == 'Yes' else 0

# Medical Conditions Section
st.header('Medical Conditions')
col1, col2 = st.columns(2)
with col1:
    input_data['Stroke'] = 1 if st.radio('Have you had a stroke?', ['No', 'Yes']) == 'Yes' else 0
    input_data['DiffWalking'] = 1 if st.radio('Difficulty walking?', ['No', 'Yes']) == 'Yes' else 0
    input_data['Asthma'] = 1 if st.radio('Do you have asthma?', ['No', 'Yes']) == 'Yes' else 0
with col2:
    input_data['KidneyDisease'] = 1 if st.radio('Kidney disease?', ['No', 'Yes']) == 'Yes' else 0
    input_data['SkinCancer'] = 1 if st.radio('Skin cancer?', ['No', 'Yes']) == 'Yes' else 0

    diabetic_status = st.selectbox('Diabetic status:', ['No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'])
    if diabetic_status == 'Yes':
        input_data['Diabetic'] = 1
    elif diabetic_status == 'No':
        input_data['Diabetic'] = 0
    else:
        input_data['Diabetic'] = 0.5  # For borderline or gestational

# Predict Button
if st.button('Predict Heart Disease Risk'):
    # Create input DataFrame and ensure correct feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[expected_features]  # Ensures model compatibility

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

    # Explanation
    st.subheader('Explanation')
    st.write("""
    This prediction is based on your input data compared to patterns learned from thousands of patient records.
    Risk factors include age, lifestyle (smoking, physical inactivity), and conditions like diabetes or stroke.
    """)

    # Recommendations
    st.subheader('Recommendations')
    st.write("""
    - Consult a healthcare provider for a full assessment
    - Maintain a healthy diet and exercise regularly
    - Monitor blood pressure, blood sugar, and cholesterol
    - Avoid smoking and limit alcohol intake
    """)
