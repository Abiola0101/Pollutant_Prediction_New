import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
import numpy as np
import pickle
import os

def combine_dataframe(df1, df2):
    combined_dataframe = pd.concat([df1, df2], axis=0)
    combined_dataframe.reset_index(drop=True, inplace=True)
    return combined_dataframe

def create_lags_no_group(df, feature, n_lags):
    for i in range(1, n_lags + 1):
        df[f'{feature}_lag{i}'] = df[feature].shift(i)
    return df

def train_model(data, start_year, n_lags, target, params=None):
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

    train_data = data[data['Reporting_Year/Année'] < start_year]
    test_data = data[data['Reporting_Year/Année'] >= start_year]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(random_state=42, **(params if params else {})))])
    pipeline.fit(X_train, y_train)

    # Evaluate model performance on the test set
    y_pred = pipeline.predict(X_test)
    metrics = {
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
        'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
        'R² Score': r2_score(y_test, y_pred)
    }

    # Specify the directory and file name where the model should be saved
    model_directory = '/home/rutholasupo/2500_Labs/model'
    model_filename = 'random_forest_model.pkl'

    # Ensure the directory exists
    os.makedirs(model_directory, exist_ok=True)

    # Save the trained model using pickle
    model_path = os.path.join(model_directory, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {model_path}")

    return pipeline, metrics

def main():
    train_path = "/home/rutholasupo/2500_Labs/data/processed/train_processed.csv"
    test_path = "/home/rutholasupo/2500_Labs/data/processed/test_processed.csv"
    combined_data_path = "/home/rutholasupo/2500_Labs/data/processed/combined_data.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    combined_df = combine_dataframe(df_train, df_test)
    
    # Save the combined data
    combined_df.to_csv(combined_data_path, index=False)
    print(f"Combined data saved to {combined_data_path}")

    start_year = 2020
    n_lags = 5
    target = 'Total_Release_Water'
    params = {'n_estimators': 100, 'max_depth': 10}

    # Train the model
    model, metrics = train_model(combined_df, start_year, n_lags, target, params)
    print("Model training complete and saved.")
    print("Metrics:\n", metrics)

if __name__ == "__main__":
    main()