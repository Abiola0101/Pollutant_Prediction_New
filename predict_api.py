import pandas as pd
import numpy as np
import category_encoders as ce
import mlflow.sklearn
import yaml
import os
import json
import argparse
import pickle  # Import the pickle module
from flask import Flask, jsonify, request
import logging
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
from config import Config

# Initialize Flask
app = Flask(__name__)

# Initialize Prometheus metrics after app is defined
metrics = PrometheusMetrics(app)

# Custom metrics
prediction_requests = Counter('model_prediction_requests_total', 'Total number of prediction requests', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')

# Configure basic logging
log_dir = os.environ.get('LOG_DIR', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, 'app.log') 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app.log')  # Log to file
    ]
)

# Create a logger for a specific module
logger = logging.getLogger(__name__)

# Define absolute project path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_PATH, 'models')
CAT_NAMES_PATH = os.path.join(PROJECT_PATH, 'configs/cat_to_name.json')

# Load category mapping
with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

# Load the configuration settings
app.config.from_object(Config)

# Access the model paths like this
MODEL_V1_PATH = app.config['MODEL_V1_PATH']
MODEL_V2_PATH = app.config['MODEL_V2_PATH']

# Load models
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return None

model_v1 = load_model(MODEL_V1_PATH)
model_v2 = load_model(MODEL_V2_PATH)

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def create_lags_no_group(self, df, feature, n_lags):
        logger.debug(f"Creating {n_lags} lags for feature: {feature}")
        for i in range(1, n_lags + 1):
            df[f'{feature}_lag{i}'] = df[feature].shift(i)
        return df

    def forecast_future_years_with_metrics(self, data, start_year, end_year, n_lags=5, target='Total_Release_Water'):
        pollutants = [target]
        additional_features = [
            'Population', 'Number_of_Employees', 'Release_to_Air(Fugitive)', 'Release_to_Air(Other_Non-Point)', 
            'Release_to_Air(Road dust)', 'Release_to_Air(Spills)', 'Release_to_Air(Stack/Point)', 
            'Release_to_Air(Storage/Handling)', 'Releases_to_Land(Leaks)', 'Releases_to_Land(Other)', 
            'Releases_to_Land(Spills)', 'Sum_of_release_to_all_media_(<1tonne)'
        ]

        # Check if all required columns are present and provide default values for missing columns
        required_columns = pollutants + additional_features + ['PROVINCE', 'Estimation_Method/Méthode destimation']
        for col in required_columns:
            if col not in data.columns:
                data[col] = 0  # Provide a default value of 0 for missing columns

        for feature in pollutants + additional_features:
            data = self.create_lags_no_group(data, feature, n_lags)

        # Drop rows with missing values
        data = data.dropna()

        # One-hot encoding
        province_encoded = pd.get_dummies(data['PROVINCE'], prefix='PROVINCE', drop_first=True, dtype=int)
        data = pd.concat([data, province_encoded], axis=1)
        estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
        data = pd.concat([data, estimation_encoded], axis=1)

        # Apply TargetEncoder
        encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination sociale de l'entreprise"])
        data = encoder.fit_transform(data, data[target])

        features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
                   [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
                   list(province_encoded.columns) + ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination sociale de l'entreprise"] + \
                   list(estimation_encoded.columns)

        if 'Region' in data.columns:
            features.append('Region')

        future_forecasts = []
        for year in range(start_year, end_year + 1):
            latest_data = data[data['Reporting_Year/Année'] == (year - 1)].copy()
            if latest_data.empty:
                continue
            latest_data['Reporting_Year/Année'] = year
            forecast_features = latest_data[features]
            latest_data[target] = self.model.predict(forecast_features)
            yearly_forecast = latest_data.groupby('PROVINCE')[[target]].sum()
            yearly_forecast['Year'] = year
            future_forecasts.append(yearly_forecast)

        if future_forecasts:
            future_forecasts = pd.concat(future_forecasts).reset_index()
        else:
            future_forecasts = pd.DataFrame()

        return future_forecasts

# Add a function to monitor resource usage in the background
def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    import psutil
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

# Flask routes
@app.route('/predictor_home', methods=['GET'])
def index():
    return f'Model V1 path: {MODEL_V1_PATH}, Model V2 path: {MODEL_V2_PATH}'

@app.route('/health_status', methods=['GET'])
def health_check():
    health = {
        "health_status": "running",
        "message": "pollution predictor is running"
    }
    return jsonify(health)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    # Get the incoming data as JSON
    data = request.get_json()

    # Check if the data is None (meaning no JSON data was provided)
    if not data:
        logger.error("No data provided in request")
        return jsonify({"error": "No JSON data provided"}), 400

    logger.info(f"Received prediction request (v1) with data: {data}")

    # Extract parameters from the input data
    start_year = data.get('start_year')
    end_year = data.get('end_year')
    n_lags = data.get('n_lags', 5)  # Default to 5 if not provided
    target = data.get('target', 'Total_Release_Water')

    # Validate that 'end_year' is provided
    if not end_year:
        logger.error("Missing 'end_year' parameter")
        return jsonify({"error": "Missing 'end_year' parameter"}), 400

    # Load the 'data' into a DataFrame for prediction
    try:
        df = pd.DataFrame(data['data'])
    except KeyError:
        logger.error("Data format is incorrect, 'data' key missing")
        return jsonify({"error": "Invalid data format, 'data' key missing"}), 400

    # Instantiate the ModelPredictor with the RandomForest model
    predictor = ModelPredictor(model=model_v1)

    # Try to forecast future years and handle any exceptions
    try:
        result = predictor.forecast_future_years_with_metrics(df, start_year, end_year, n_lags, target)
    except KeyError as e:
        logger.error(f"KeyError during prediction: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

    # Return the result as a JSON response
    return jsonify({
        "success": True,
        "prediction": result.to_dict()
    })

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract parameters from input data
    start_year = data.get('start_year')
    end_year = data.get('end_year')
    n_lags = data.get('n_lags', 5)
    target = data.get('target', 'Total_Release_Water')

    # Load data from the incoming request
    df = pd.DataFrame(data['data'])

    if not end_year:
        return jsonify({"error": "Missing 'end_year' parameter"}), 400

    # Instantiate model predictor with the GradientBoosting model
    predictor = ModelPredictor(model=model_v2)

    try:
        result = predictor.forecast_future_years_with_metrics(df, start_year, end_year, n_lags, target)
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "prediction": result.to_dict()
    })

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
