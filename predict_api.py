import pandas as pd
import pickle
import numpy as np
import category_encoders as ce
import mlflow.sklearn
import yaml
import argparse
import os
import json

# import jsonify and request from flask
from flask import Flask, jsonify, request   

# initialize flask
app = Flask(__name__)

# define absolute project path
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# load category mapping
CAT_NAMES_PATH = os.path.join(PROJECT_PATH, 'configs/cat_to_name.json')

MODELS_DIR = os.path.join(PROJECT_PATH, 'models')

with open(CAT_NAMES_PATH, 'r') as f:
    cat_to_name = json.load(f)

# specify the model paths
MODEL_V1_PATH = os.path.join(MODELS_DIR, 'RandomForest_random_forest_model.pkl')
MODEL_V2_PATH = os.path.join(MODELS_DIR, 'GradientBoosting_random_forest_model.pkl')

# Load models with error handling
try:
    with open(MODEL_V1_PATH, "rb") as f:
        model_v1 = pickle.load(f)
    print(f"RandomForest model loaded from {MODEL_V1_PATH}")
except FileNotFoundError:
    print(f"File not found: {MODEL_V1_PATH}")
    model_v1 = None

try:
    with open(MODEL_V2_PATH, "rb") as f:
        model_v2 = pickle.load(f)
    print(f"GradientBoosting model loaded from {MODEL_V2_PATH}")
except FileNotFoundError:
    print(f"File not found: {MODEL_V2_PATH}")
    model_v2 = None

def create_lags_no_group(df, feature, n_lags):
    for i in range(1, n_lags + 1):
        df[f'{feature}_lag{i}'] = df[feature].shift(i)
    return df

def forecast_future_years_with_metrics(data, start_year, end_year, n_lags=5, target='Total_Release_Water'):
    pollutants = [target]
    additional_features = [
        'Population', 'Number_of_Employees', 'Release_to_Air(Fugitive)', 'Release_to_Air(Other_Non-Point)', 
        'Release_to_Air(Road dust)', 'Release_to_Air(Spills)', 'Release_to_Air(Stack/Point)', 
        'Release_to_Air(Storage/Handling)', 'Releases_to_Land(Leaks)', 'Releases_to_Land(Other)', 
        'Releases_to_Land(Spills)', 'Sum_of_release_to_all_media_(<1tonne)'
    ]
    
    for feature in pollutants + additional_features:
        data = create_lags_no_group(data, feature, n_lags)
    
    data = data.dropna()
    province_encoded = pd.get_dummies(data['PROVINCE'], prefix='PROVINCE', drop_first=True, dtype=int)
    data = pd.concat([data, province_encoded], axis=1)
    estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
    data = pd.concat([data, estimation_encoded], axis=1)

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
        latest_data[target] = model_v1.predict(forecast_features)  # Use model_v1 or model_v2 as needed
        yearly_forecast = latest_data.groupby('PROVINCE')[[target]].sum()
        yearly_forecast['Year'] = year
        future_forecasts.append(yearly_forecast)

    if future_forecasts:
        future_forecasts = pd.concat(future_forecasts).reset_index()
    else:
        future_forecasts = pd.DataFrame()

    return future_forecasts

# create decorator with specific routes, to find specific things required for code to run
@app.route('/predictor_home', methods=['GET'])
def home():
    return "Welcome to the predictor home page"

@app.route('/health_status', methods=['GET'])
def health_check():
    health = {
        "health_status": "running",
        "message": "pollution predictor is running"
    }
    return jsonify(health)

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    # Get the JSON data from the request
    data = request.get_json()

    # Check if the data is correctly received
    if data is None:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract parameters
    start_year = data.get('start_year')
    end_year = data.get('end_year')
    n_lags = data.get('n_lags')
    target = data.get('target')

    # Debugging: Print data to verify if it's being received correctly
    print("Received data:", data)

    # Ensure that 'end_year' is provided
    if end_year is None:
        return jsonify({"error": "Missing 'end_year' parameter"}), 400

    # Call the forecast function
    result = forecast_future_years_with_metrics(data, start_year, end_year, n_lags=n_lags, target=target)

    return jsonify({
        "success": True,
        "prediction": result.to_dict()
    })

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    # get the data from the POST request
    return "Nothing"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9999, debug=True)