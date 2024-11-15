import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, time

# Set page config
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    with open('sleep_quality_model.pkl', 'rb') as f:
        model = pickle.load(f)
    if isinstance(model, dict):
        st.error("Loaded model is a dictionary, not a trained model object. Please verify the model file.")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()


# Title
st.title("Sleep Quality Prediction App")
st.write("Enter your details below to predict sleep quality")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    # Time inputs
    bedtime = st.time_input("Bedtime", value=time(23, 0))
    wakeup_time = st.time_input("Wake-up Time", value=time(7, 0))
    
    # Convert times to decimal hours
    bedtime_decimal = bedtime.hour + bedtime.minute/60
    wakeup_decimal = wakeup_time.hour + wakeup_time.minute/60
    
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
    calories_burned = st.number_input("Calories Burned", min_value=0, max_value=5000, value=2500)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0)

with col2:
    # Categorical inputs
    gender = st.selectbox("Gender", options=['f', 'm'])
    
    physical_activity = st.selectbox(
        "Physical Activity Level",
        options=['low', 'medium', 'high']
    )
    
    dietary_habits = st.selectbox(
        "Dietary Habits",
        options=['healthy', 'medium', 'unhealthy']
    )
    
    sleep_disorders = st.selectbox(
        "Sleep Disorders",
        options=['no', 'yes']
    )
    
    medication_usage = st.selectbox(
        "Medication Usage",
        options=['no', 'yes']
    )

# Predict button
if st.button("Predict Sleep Quality"):
    # Create a dictionary with the input values
    input_data = {
        'Age': age,
        'Bedtime': bedtime_decimal,
        'Wake-up Time': wakeup_decimal,
        'Daily Steps': daily_steps,
        'Calories Burned': calories_burned,
        'Sleep Duration': sleep_duration,
        'Gender': gender,
        'Physical Activity Level': physical_activity,
        'Dietary Habits': dietary_habits,
        'Sleep Disorders': sleep_disorders,
        'Medication Usage': medication_usage
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Process categorical variables
    categorical_df = pd.get_dummies(
        input_df[feature_names['categorical_features']], 
        drop_first=True
    )
    
    # Combine numerical and categorical features
    numerical_df = input_df[feature_names['numerical_features']]
    processed_input = pd.concat([numerical_df, categorical_df], axis=1)
    
    # Ensure all expected columns are present
    for col in feature_names['encoded_features']:
        if col not in processed_input.columns:
            processed_input[col] = 0
    
    # Reorder columns to match training data
    processed_input = processed_input[feature_names['encoded_features']]
    
    # Scale the features
    scaled_input = scaler.transform(processed_input)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    
    # Display prediction
    st.success(f"Predicted Sleep Quality Score: {prediction:.2f}/10")
    
    # Provide interpretation
    if prediction >= 8:
        quality = "Excellent"
        color = "green"
    elif prediction >= 6:
        quality = "Good"
        color = "blue"
    elif prediction >= 4:
        quality = "Fair"
        color = "orange"
    else:
        quality = "Poor"
        color = "red"
    
    st.markdown(f"<h3 style='color: {color}'>Sleep Quality Rating: {quality}</h3>", 
                unsafe_allow_html=True)
    
    # Additional recommendations based on input values
    st.subheader("Recommendations:")
    recommendations = []
    
    if sleep_duration < 7:
        recommendations.append("Consider increasing your sleep duration to at least 7 hours.")
    if daily_steps < 7000:
        recommendations.append("Try to increase your daily steps to at least 7,000 for better sleep quality.")
    if physical_activity == 'low':
        recommendations.append("Increasing your physical activity level could improve your sleep quality.")
    if dietary_habits == 'unhealthy':
        recommendations.append("Improving your dietary habits could have a positive impact on your sleep.")
    
    for rec in recommendations:
        st.write("• " + rec)

# Add footer with additional information
st.markdown("---")
st.markdown("""
    **Note:** This prediction is based on a machine learning model trained on historical sleep data.
    The sleep quality score ranges from 1-10, where 10 represents the best possible sleep quality.
""")