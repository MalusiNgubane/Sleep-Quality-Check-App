# Sleep-Quality Predictor
The Health and Sleep Statistics dataset provides a rich source of information about individuals’ sleep habits and physical activity. However, it is necessary to explore this data to uncover patterns and correlations between these lifestyle factors and their potential impact on overall health. 

Main Base: A machine learning application that predicts sleep quality based on various lifestyle factors.

1. Clone the repository:
```bash
git clone https://github.com/MalusiNgubane/Sleep-Quality-Check-App.git
cd Sleep-Quality-Check-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Main Streamlit application
- `src/utils.py`: Utility functions for data processing
- `models/`: Directory containing trained models and scalers
- `requirements.txt`: Project dependencies

## Deploy on Streamlit Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy the app

## Model Information
The sleep quality prediction model is trained on lifestyle and health data to predict sleep quality on a scale of 1-10.
