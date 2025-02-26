import pandas as pd
import pickle
import numpy as np
import category_encoders as ce

def forecast_future_years_with_metrics(data, start_year, end_year, n_lags=5, target='Total_Release_Water', model=None):
    def create_lags_no_group(df, feature, n_lags):
        for i in range(1, n_lags + 1):
            df[f'{feature}_lag{i}'] = df[feature].shift(i)
        return df

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
    estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode_destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
    data = pd.concat([data, estimation_encoded], axis=1)

    encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"])
    data = encoder.fit_transform(data, data[target])

    features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
               [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
               list(province_encoded.columns) + ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"] + \
               list(estimation_encoded.columns)

    if 'Region' in data.columns:
        features += ['Region']

    future_forecasts = []
    for year in range(start_year, end_year + 1):
        latest_data = data[data['Reporting_Year/Année'] == (year - 1)].copy()
        if latest_data.empty:
            continue
        latest_data['Reporting_Year/Année'] = year
        forecast_features = latest_data[features]
        latest_data[target] = model.predict(forecast_features)
        yearly_forecast = latest_data.groupby('PROVINCE')[[target]].sum()
        yearly_forecast['Year'] = year
        future_forecasts.append(yearly_forecast)

    if future_forecasts:
        future_forecasts = pd.concat(future_forecasts).reset_index()
    else:
        future_forecasts = pd.DataFrame()

    return future_forecasts

def main():
    # Load the trained model
    with open('/home/rutholasupo/2500_Labs/model/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the feature-engineered dataset
    data_path = "/home/rutholasupo/2500_Labs/data/processed/combined_data.csv"
    combined_df = pd.read_csv(data_path)

    start_year = 2020
    end_year = 2023
    n_lags = 5
    target = 'Total_Release_Water'

    # Run forecasting
    forecasts = forecast_future_years_with_metrics(combined_df, start_year, end_year, n_lags, target, model)
    print("Forecasts:\n", forecasts)

if __name__ == "__main__":
    main()
