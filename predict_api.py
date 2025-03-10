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

# load checkpoint



# specify the model path
MODEL_V1_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
# MODEL_PATH = os.path.join(PROJECT_PATH, 'models/model.pkl')

def forecast_future_years_with_metrics(self, data, start_year, end_year, n_lags=5, target='Total_Release_Water'):
    pollutants = [target]
    additional_features = [
        'Population', 'Number_of_Employees', 'Release_to_Air(Fugitive)', 'Release_to_Air(Other_Non-Point)', 
        'Release_to_Air(Road dust)', 'Release_to_Air(Spills)', 'Release_to_Air(Stack/Point)', 
        'Release_to_Air(Storage/Handling)', 'Releases_to_Land(Leaks)', 'Releases_to_Land(Other)', 
        'Releases_to_Land(Spills)', 'Sum_of_release_to_all_media_(<1tonne)'
    ]
    
    for feature in pollutants + additional_features:
        data = self.create_lags_no_group(data, feature, n_lags)
    
    data = data.dropna()
    province_encoded = pd.get_dummies(data['PROVINCE'], prefix='PROVINCE', drop_first=True, dtype=int)
    data = pd.concat([data, province_encoded], axis=1)
    estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode_destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
    data = pd.concat([data, estimation_encoded], axis=1)

    encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"])
    data = encoder.fit_transform(data, data[target])

    features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
               [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
               list(province_encoded.columns) + ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"] + \
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

# create decorator with specific routes, to find specific things required for code to run
@app.route('/predictor_home', methods=['GET'])
def home():

    return "Welcome to the predictor home page"
    # return "Welcome to the predictor home page"


@app.route('/health_status', methods=['GET'])
def health_check():

    health ={
        "health_status": "runing",
        "message": "pollution predictor is running"
    }
    return jsonify(health)


    # return "Predictor is healthy"

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    # get the data from the POST request
    if not request.json:
        return "No data provided"
    data = request.json

    if 'data' not in data:
        return "No data provided"
    
    data = data['data']
    start_year = data['start_year']
    end_year = data['end_year']
    n_lags = data['n_lags']
    target = data['target']

    result = forecast_future_years_with_metrics(data, start_year, end_year, n_lags=5, target='Total_Release_Water')

    return jsonify({
        "success": True,
        "prediction": result

    })


@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    # get the data from the POST request
    return "Nothing"



    # # get the data from the POST request
    # data = request.get_json(force=True)

    # # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    # # make prediction
    # future_forecasts = forecast_future_years_with_metrics(data_df, 2018, 2020)

    # # convert prediction to list
    # output = future_forecasts.to_dict(orient='records')

    # return jsonify(output)


if __name__ == '__main__':
    app.run(host= '127.0.0.1', port = '9999', debug = True)