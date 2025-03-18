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

# Initialize Flask
app = Flask(__name__)

# Define absolute project path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_PATH, 'models')
CAT_NAMES_PATH = os.path.join(PROJECT_PATH, 'configs/cat_to_name.json')

# Load category mapping
with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

# Specify the model paths (example models)
#MODEL_V1_PATH = os.path.join(MODELS_DIR, 'RandomForest_random_forest_model.pkl')
#MODEL_V2_PATH = os.path.join(MODELS_DIR, 'GradientBoosting_random_forest_model.pkl')

#MODEL_V1_PATH = '/home/abi_norquest_ml/2500_Labs/model/RandomForest_random_forest_model.pkl'
#MODEL_V2_PATH = '/home/abi_norquest_ml/2500_Labs/model/GradientBoosting_random_forest_model.pkl'


#MODEL_V1_PATH = '/home/abiola/Pollutant_Prediction_New/model/RandomForest_random_forest_model.pkl'
#MODEL_V2_PATH = '/home/abiola/Pollutant_Prediction_New/model/GradientBoosting_random_forest_model.pkl'

MODEL_V1_PATH = '/app/model/RandomForest_random_forest_model.pkl'
MODEL_V2_PATH = '/app/model/GradientBoosting_random_forest_model.pkl'


# Load models
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"File not found: {model_path}")
        return None

model_v1 = load_model(MODEL_V1_PATH)
model_v2 = load_model(MODEL_V2_PATH)

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def create_lags_no_group(self, df, feature, n_lags):
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

# Flask routes
@app.route('/predictor_home', methods=['GET'])
def home():
    app_info = {
        "API_Name": "Pollutant predictor",
        "API Description": "This API takes parameters from user and returns a prediction of amount of pollutants",
        "version": "v1.0",
        "endpoints": {
            "/predictor_home": "Home page",
            "/health_status": "Displays the health status of the API",
            "v1/predict": "This version of API is based on RandomForest model",
            "v2/predict": "This version of API is based on GradientBoost model"
        },
        "Input format": {
            "start_year": 2020,
            "end_year": 2023,
            "n_lags": 4,
            "target": "Total_Release_Water",
            "data": [
                {
                    "PROVINCE": "SomeProvince",
                    "City": "SomeCity",
                    "Facility_Name/Installation": "SomeFacility",
                    "Total_Release_Water": 1000,
                    "Population": 100000,
                    "Number_of_Employees": 500,
                    "Release_to_Air(Fugitive)": 200
                },
                {
                    "PROVINCE": "AnotherProvince",
                    "City": "AnotherCity",
                    "Facility_Name/Installation": "AnotherFacility",
                    "Total_Release_Water": 1500,
                    "Population": 120000,
                    "Number_of_Employees": 600,
                    "Release_to_Air(Fugitive)": 250
                }
            ]
        },
        "example request": {
            'curl -X POST "http://127.0.0.1:5050/v1/predict" \\': '',
            '-H "Content-Type: application/json" \\': '',
            '-d \'{': '',
            '"start_year": 2023,': '',
            '"end_year": 2023,': '',
            '"n_lags": 2,': '',
            '"target": "Total_Release_Water",': '',
            '"data": [': '',
            '{': '',
            '"Reporting_Year/Année": 2022,': '',
            '"Population": 50000,': '',
            '"Number_of_Employees": 100,': '',
            '"Release_to_Air(Fugitive)": 120.5,': '',
            '"Release_to_Air(Other_Non-Point)": 60.2,': '',
            '"Release_to_Air(Road dust)": 24,': '',
            '"Release_to_Air(Spills)": 110.2,': '',
            '"Release_to_Air(Stack/Point)": 16.5,': '',
            '"Release_to_Air(Storage/Handling)": 45.2,': '',
            '"Releases_to_Land(Leaks)": 111.2,': '',
            '"Releases_to_Land(Other)": 24,': '',
            '"Releases_to_Land(Spills)": 36,': '',
            '"Sum_of_release_to_all_media_(<1tonne)": 16.6,': '',
            '"PROVINCE": "ON",': '',
            '"Estimation_Method/Méthode destimation": "Calculated",': '',
            '"City": "Toronto",': '',
            '"Facility_Name/Installation": "ABC Plant",': '',
            '"NAICS Title/Titre_Code_SCIAN": "Chemical Manufacturing",': '',
            '"NAICS/Code_SCIAN": 325110,': '',
            '"Company_Name/Dénomination sociale de l\'entreprise": "XYZ Corp"': '',
            '}': '',
            ']': '',
            '}\'': ''
        },
        "example response": {
            "success": True,
            "prediction": {
                "PROVINCE": "ON",
                "Total_Release_Water": 1000,
                "Year": 2023
            }
        }
    }

    return jsonify(app_info)


@app.route('/health_status', methods=['GET'])
def health_check():
    health = {
        "health_status": "running",
        "message": "pollution predictor is running"
    }
    return jsonify(health)


@app.route('/v1/predict', methods=['POST'])
def predict_v1():
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

    # Instantiate model predictor with the RandomForest model
    predictor = ModelPredictor(model=model_v1)

    try:
        result = predictor.forecast_future_years_with_metrics(df, start_year, end_year, n_lags, target)
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)})

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
        return jsonify({"success": False, "error": str(e)})

    return jsonify({
        "success": True,
        "prediction": result.to_dict()
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Prediction API")
    parser.add_argument('--config', type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument('--combined_data_path', type=str, required=True, help="Path to the data file")
    parser.add_argument('--end_year', type=int, required=True, help="End year for forecasting")

    args = parser.parse_args()

    # Set config parameters for the app
    app.config['CONFIG'] = args.config
    app.config['COMBINED_DATA_PATH'] = args.combined_data_path
    app.config['END_YEAR'] = args.end_year

    app.run(host='0.0.0.0', port=9000, debug=True)