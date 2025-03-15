import argparse
import pandas as pd
import pickle
import numpy as np
import category_encoders as ce
import mlflow.sklearn
import yaml
import os
from mlflow.models.signature import infer_signature

class ModelPredictor:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = self.load_model()

    def load_model(self):
        run_id = self.config['run_id']
        model_uri = f"runs:/{run_id}/model"
        return mlflow.sklearn.load_model(model_uri)

    def create_lags_no_group(self, df, feature, n_lags):
        for i in range(1, n_lags + 1):
            df[f'{feature}_lag{i}'] = df[feature].shift(i)
        return df

    def forecast_future_years_with_metrics(self, data, start_year, end_year, n_lags, target):
        pollutants = [target]
        additional_features = [
            'Population', 'Number_of_Employees', 'Release_to_Air(Fugitive)', 'Release_to_Air(Other_Non-Point)', 
            'Release_to_Air(Road dust)', 'Release_to_Air(Spills)', 'Release_to_Air(Stack/Point)', 
            'Release_to_Air(Storage/Handling)', 'Releases_to_Land(Leaks)', 'Releases_to_Land(Other)', 
            'Releases_to_Land(Spills)', 'Sum_of_release_to_all_media_(<1tonne)'
        ]
        
        # Create lags for relevant features
        for feature in pollutants + additional_features:
            data = self.create_lags_no_group(data, feature, n_lags)
        
        # Drop rows with missing values due to lagging
        data = data.dropna()

        # One-hot encode 'PROVINCE' and 'Estimation_Method/Méthode_destimation'
        province_encoded = pd.get_dummies(data['PROVINCE'], prefix='PROVINCE', drop_first=True, dtype=int)
        data = pd.concat([data, province_encoded], axis=1)
        estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode_destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
        data = pd.concat([data, estimation_encoded], axis=1)

        # Apply TargetEncoder for categorical features
        encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"])
        data = encoder.fit_transform(data, data[target])

        # Define the feature list for the model
        features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
                   [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
                   list(province_encoded.columns) + ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"] + \
                   list(estimation_encoded.columns)

        if 'Region' in data.columns:
            features.append('Region')

        # Forecast for each year
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

    def main(self):
        # Load dataset path and other parameters from the config
        data_path = self.config['combined_data_path']
        combined_df = pd.read_csv(data_path)

        # Retrieve configuration parameters
        start_year = self.config['start_year']
        n_lags = self.config['n_lags']
        target = self.config['target']

        # Get the end year for forecasting from the user input or use default
        end_year = self.config.get('end_year', 2023)  # Use end_year from config or default to 2023 if not provided

        # Forecast future years
        forecasts = self.forecast_future_years_with_metrics(combined_df, start_year, end_year, n_lags, target)
        print("Forecasts:\n", forecasts)

        # Optionally, save the forecasts to a file
        forecast_output_path = self.config.get('forecast_output_path', 'forecast_results.csv')
        forecasts.to_csv(forecast_output_path, index=False)
        print(f"Forecasts saved to {forecast_output_path}")

if __name__ == "__main__":
    # Parse arguments for configuration file
    parser = argparse.ArgumentParser(description="Predict future metrics using a trained model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Instantiate and run the predictor
    predictor = ModelPredictor(args.config)
    predictor.main()
