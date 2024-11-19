import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9ff;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, time

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained machine learning model, scaler, and feature names."""
    try:
        with open('sleep_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        if not hasattr(model, 'predict'):
            st.error("Loaded model is not a trained model object. Please verify the model file.")

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model or preprocessing objects: {e}")
        return None, None, None

# Load the model, scaler, and feature names
model, scaler, feature_names = load_model()

# Display app title and instructions
st.title("🌙Sleep Quality Prediction App")
st.write("Enter your details below to predict sleep quality.")

# Create two columns for user inputs
col1, col2 = st.columns(2)

# Collect user inputs
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bedtime = st.time_input("Bedtime", value=time(23, 0))
    wakeup_time = st.time_input("Wake-up Time", value=time(7, 0))
    bedtime_decimal = bedtime.hour + bedtime.minute / 60
    wakeup_decimal = wakeup_time.hour + wakeup_time.minute / 60
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
    calories_burned = st.number_input("Calories Burned", min_value=0, max_value=5000, value=2500)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0)

with col2:
    gender = st.selectbox("Gender", options=['f', 'm'])
    physical_activity = st.selectbox("Physical Activity Level", options=['low', 'medium', 'high'])
    dietary_habits = st.selectbox("Dietary Habits", options=['healthy', 'medium', 'unhealthy'])
    sleep_disorders = st.selectbox("Sleep Disorders", options=['no', 'yes'])
    medication_usage = st.selectbox("Medication Usage", options=['no', 'yes'])

# Predict button
if st.button("Predict Sleep Quality"):
    # Validate inputs
    if sleep_duration <= 0:
        st.error("Sleep Duration must be greater than 0.")
    else:
        # Prepare input data as a dictionary
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

        # Process categorical features
        try:
            categorical_df = pd.get_dummies(
                input_df[feature_names['categorical_features']],
                drop_first=True
            )
        except KeyError:
            st.error("Error processing categorical features. Please check feature names.")
            categorical_df = pd.DataFrame()

        # Combine numerical and categorical features
        numerical_df = input_df[feature_names['numerical_features']]
        processed_input = pd.concat([numerical_df, categorical_df], axis=1)

        # Add missing columns if necessary
        for col in feature_names['encoded_features']:
            if col not in processed_input.columns:
                processed_input[col] = 0

        # Reorder columns to match the training data
        processed_input = processed_input[feature_names['encoded_features']]

        # Scale the input features
        scaled_input = scaler.transform(processed_input)

        # Ensure input shape matches the model's requirements
        if scaled_input.shape[1] != model.n_features_in_:
            st.error("Input features do not match the model's expected features. Please check your preprocessing.")
        else:
            # Make a prediction
            prediction = model.predict(scaled_input)[0]

            # Display prediction
            st.success(f"Predicted Sleep Quality Score: {prediction:.2f}/10")

            # Interpret the prediction
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

            st.markdown(f"<h3 style='color: {color}'>Sleep Quality Rating: {quality}</h3>", unsafe_allow_html=True)

            # Provide recommendations
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

            # Add a radar chart to compare with benchmarks
    st.subheader("Interactive Visualization: Sleep Quality Comparison")
    benchmarks = {
        "Sleep Duration (hours)": 8,
        "Daily Steps": 10000,
        "Calories Burned": 2500,
        "Physical Activity Level (score)": 2,  # Low=1, Medium=2, High=3
        "Sleep Quality Score": 8
    }
    user_values = {
        "Sleep Duration (hours)": sleep_duration,
        "Daily Steps": daily_steps,
        "Calories Burned": calories_burned,
        "Physical Activity Level (score)": 1 if physical_activity == 'low' else 2 if physical_activity == 'medium' else 3,
        "Sleep Quality Score": prediction
    }

    # Convert data for radar chart
    categories = list(benchmarks.keys())
    user_scores = list(user_values.values())
    benchmark_scores = list(benchmarks.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_scores,
        theta=categories,
        fill='toself',
        name='User'
    ))
    fig.add_trace(go.Scatterpolar(
        r=benchmark_scores,
        theta=categories,
        fill='toself',
        name='Ideal Benchmarks'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(benchmark_scores) + 2])
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

# Footer section
st.markdown("---")
st.markdown("""
    **Note:** This prediction is based on a machine learning model trained on historical sleep data.
    The sleep quality score ranges from 1-10, where 10 represents the best possible sleep quality.
""")
