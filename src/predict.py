import pandas as pd
import pickle
import numpy as np
import category_encoders as ce
import mlflow.sklearn
import yaml

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

    def main(self):
        data_path = self.config['combined_data_path']
        combined_df = pd.read_csv(data_path)

        start_year = self.config['start_year']
        n_lags = self.config['n_lags']
        target = self.config['target']

        end_year = input(f"Enter end year for forecasting (default: 2023): ") or 2023
        end_year = int(end_year)

        forecasts = self.forecast_future_years_with_metrics(combined_df, start_year, end_year, n_lags, target)
        print("Forecasts:\n", forecasts)

if __name__ == "__main__":
    predictor = ModelPredictor('/home/rutholasupo/2500_Labs/configs/train_config.yaml')
    predictor.main()