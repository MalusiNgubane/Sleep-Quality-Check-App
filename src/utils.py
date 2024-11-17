import pickle
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Any

def load_model() -> Tuple[Any, Any, Dict]:
    """
    Load the model and preprocessing objects from pickle files.
    
    Returns:
        tuple: (model, scaler, feature_names)
    """
    try:
        with open('models/sleep_quality_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Required model files not found: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def process_input_data(input_data: Dict, feature_names: Dict) -> pd.DataFrame:
    """
    Process input data for prediction.
    
    Args:
        input_data: Dictionary containing user input
        feature_names: Dictionary containing feature names
        
    Returns:
        DataFrame: Processed input ready for prediction
    """
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
    return processed_input[feature_names['encoded_features']]

def get_sleep_quality_rating(prediction: float) -> Tuple[str, str]:
    """
    Get sleep quality rating and color based on prediction.
    
    Args:
        prediction: Predicted sleep quality score
        
    Returns:
        tuple: (quality_rating, color)
    """
    if prediction >= 8:
        return "Excellent", "green"
    elif prediction >= 6:
        return "Good", "blue"
    elif prediction >= 4:
        return "Fair", "orange"
    return "Poor", "red"

def generate_recommendations(sleep_duration: float, daily_steps: int,
                           physical_activity: str, dietary_habits: str) -> list:
    """
    Generate recommendations based on user input.
    
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    if sleep_duration < 7:
        recommendations.append("Consider increasing your sleep duration to at least 7 hours.")
    if daily_steps < 7000:
        recommendations.append("Try to increase your daily steps to at least 7,000 for better sleep quality.")
    if physical_activity == 'low':
        recommendations.append("Increasing your physical activity level could improve your sleep quality.")
    if dietary_habits == 'unhealthy':
        recommendations.append("Improving your dietary habits could have a positive impact on your sleep.")
    
    return recommendations
