import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
import pickle
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
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding: 2rem !important;
    }
    [data-testid="stSidebar"] {
        display: none
    }
    section[data-testid="stSidebarContent"] {
        display: none
    }
    button[kind="header"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

def create_gauge(value, title, min_val, max_val, target_val, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#4F46E5"},
            'steps': [
                {'range': [min_val, target_val * 0.5], 'color': "lightgray"},
                {'range': [target_val * 0.5, target_val * 0.8], 'color': "gray"},
                {'range': [target_val * 0.8, max_val], 'color': "#E5E7EB"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target_val
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def get_sleep_recommendations(prediction, input_data):
    recommendations = []
    detailed_advice = []
    
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
    
    if input_data['Sleep Duration'] < 7:
        recommendations.append("Increase sleep duration to 7-9 hours")
        detailed_advice.extend([
            "Set a consistent bedtime and wake-up schedule",
            "Gradually adjust bedtime by 15 minutes earlier each week",
            "Use bedtime reminders on your phone"
        ])
    
    if input_data['Daily Steps'] < 7000:
        recommendations.append("Increase daily physical activity")
        detailed_advice.extend([
            "Aim for 7,000-10,000 steps daily",
            "Take walking breaks during work",
            "Use stairs instead of elevator"
        ])
    
    if input_data['Physical Activity Level'].lower() == 'low':
        recommendations.append("Enhance exercise routine")
        detailed_advice.extend([
            "Include 150 minutes of moderate exercise weekly",
            "Try morning exercises for better sleep",
            "Avoid intense workouts 2-3 hours before bedtime"
        ])
    
    if input_data['Dietary Habits'].lower() == 'unhealthy':
        recommendations.append("Improve dietary habits")
        detailed_advice.extend([
            "Limit caffeine after 2 PM",
            "Avoid heavy meals 3 hours before bedtime",
            "Include sleep-promoting foods (cherries, nuts, whole grains)"
        ])
    
    if input_data['Age'] > 60:
        detailed_advice.extend([
            "Consider shorter afternoon naps (20-30 minutes)",
            "Increase exposure to natural daylight",
            "Practice gentle evening stretches"
        ])
    
    if input_data['Sleep Disorders'].lower() == 'yes':
        recommendations.append("Consult healthcare provider about sleep disorders")
        detailed_advice.append("Keep a detailed sleep diary for medical consultation")
    
    bedtime = input_data['Bedtime']
    if bedtime < 21 or bedtime > 23:
        recommendations.append("Adjust bedtime schedule")
        detailed_advice.extend([
            "Aim for bedtime between 9-11 PM",
            "Create a relaxing bedtime routine",
            "Dim lights 1-2 hours before bed"
        ])
    
    return {
        'quality': quality,
        'color': color,
        'recommendations': recommendations,
        'detailed_advice': detailed_advice
    }

def display_recommendations(st, results):
    st.markdown(
        f"<h3 style='color: {results['color']}'>Sleep Quality Rating: {results['quality']}</h3>",
        unsafe_allow_html=True
    )
    
    st.subheader("Key Recommendations:")
    for rec in results['recommendations']:
        st.write(f"• {rec}")
        
    st.subheader("Detailed Action Steps:")
    for advice in results['detailed_advice']:
        st.write(f"• {advice}")

@st.cache_resource
def load_model():
    try:
        with open('sleep_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        if not hasattr(model, 'predict'):
            st.error("Loaded model is not a trained model object.")
            return None, None, None

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model or preprocessing objects: {e}")
        return None, None, None

def main():
    model, scaler, feature_names = load_model()

    st.title("🌙Sleep Quality Prediction App")
    st.write("Enter your details below to predict sleep quality.")

    col1, col2 = st.columns(2)

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

    if st.button("Predict Sleep Quality"):
        if sleep_duration <= 0:
            st.error("Sleep Duration must be greater than 0.")
        else:
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

            input_df = pd.DataFrame([input_data])
            
            try:
                categorical_df = pd.get_dummies(
                    input_df[feature_names['categorical_features']],
                    drop_first=True
                )
                numerical_df = input_df[feature_names['numerical_features']]
                processed_input = pd.concat([numerical_df, categorical_df], axis=1)

                for col in feature_names['encoded_features']:
                    if col not in processed_input.columns:
                        processed_input[col] = 0

                processed_input = processed_input[feature_names['encoded_features']]
                scaled_input = scaler.transform(processed_input)

                if scaled_input.shape[1] != model.n_features_in_:
                    st.error("Input features do not match the model's expected features.")
                else:
                    prediction = min(model.predict(scaled_input)[0], 10)
                    st.success(f"Predicted Sleep Quality Score: {prediction:.2f}/10")

                    results = get_sleep_recommendations(prediction, input_data)
                    display_recommendations(st, results)

                    gauge_col1, gauge_col2 = st.columns(2)

                    with gauge_col1:
                        calories_gauge = create_gauge(
                            calories_burned,
                            "Daily Calories Burned",
                            0,
                            5000,
                            2500,
                            "cal"
                        )
                        st.plotly_chart(calories_gauge, use_container_width=True)

                    with gauge_col2:
                        sleep_gauge = create_gauge(
                            sleep_duration,
                            "Sleep Duration",
                            0,
                            12,
                            8,
                            "hrs"
                        )
                        st.plotly_chart(sleep_gauge, use_container_width=True)

                    st.subheader("Interactive Visualization: Sleep Quality Comparison")
                    benchmarks = {
                        "Sleep Duration (hours)": 8,
                        "Daily Steps": 10000,
                        "Calories Burned": 2500,
                        "Physical Activity Level (score)": 2,
                        "Sleep Quality Score": 8
                    }
                    user_values = {
                        "Sleep Duration (hours)": sleep_duration,
                        "Daily Steps": daily_steps,
                        "Calories Burned": calories_burned,
                        "Physical Activity Level (score)": 1 if physical_activity == 'low' else 2 if physical_activity == 'medium' else 3,
                        "Sleep Quality Score": prediction
                    }

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

            except Exception as e:
                st.error(f"Error processing input data: {e}")

    st.markdown("---")
    st.markdown("""
        **Note:** This prediction is based on a machine learning model trained on historical sleep data.
        The sleep quality score ranges from 1-10, where 10 represents the best possible sleep quality.
    """)

if __name__ == "__main__":
    main()
