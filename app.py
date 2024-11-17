import streamlit as st
from datetime import time
from utils import (
    load_model,
    process_input_data,
    get_sleep_quality_rating,
    generate_recommendations
)

# Set page config
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

# Load the saved model and preprocessing objects
@st.cache_resource
def get_model():
    return load_model()

model, scaler, feature_names = get_model()

# Title
st.title("Sleep Quality Prediction App")
st.write("Enter your details below to predict sleep quality")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bedtime = st.time_input("Bedtime", value=time(23, 0))
    wakeup_time = st.time_input("Wake-up Time", value=time(7, 0))
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
    # Create input data dictionary
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
    
    # Process input data
    processed_input = process_input_data(input_data, feature_names)
    
    # Scale the features
    scaled_input = scaler.transform(processed_input)

with st.spinner('Analyzing your sleep quality...'):
    # Show a progress bar
    progress_bar = st.progress(0)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    progress_bar.progress(50)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    
    # Display prediction
    st.success(f"Predicted Sleep Quality Score: {prediction:.2f}/10")
    
    # Get and display quality rating
    quality, color = get_sleep_quality_rating(prediction)
    st.markdown(f"<h3 style='color: {color}'>Sleep Quality Rating: {quality}</h3>", 
                unsafe_allow_html=True)
    progress_bar.progress(75)
    
    # Display recommendations
    st.subheader("Recommendations:")
    recommendations = generate_recommendations(
        sleep_duration, daily_steps, physical_activity, dietary_habits
    )
    for rec in recommendations:
        st.write("• " + rec)
    
    progress_bar.progress(100)
    # Optional: remove progress bar after completion
    progress_bar.empty()

# Add footer with additional information
st.markdown("---")
st.markdown("""
    **Note:** This prediction is based on a machine learning model trained on historical sleep data.
    The sleep quality score ranges from 1-10, where 10 represents the best possible sleep quality.
""")
