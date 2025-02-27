import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import os
import numpy as np
import pickle
import subprocess
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# Load training configuration from YAML file
config_path = '/home/rutholasupo/2500_Labs/configs/train_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Enable MLflow autologging
mlflow.sklearn.autolog()

def combine_dataframe(df1, df2):
    combined_dataframe = pd.concat([df1, df2], axis=0)
    combined_dataframe.reset_index(drop=True, inplace=True)
    return combined_dataframe

def create_lags_no_group(df, feature, n_lags):
    for i in range(1, n_lags + 1):
        df[f'{feature}_lag{i}'] = df[feature].shift(i)
    return df

def train_model(data, start_year, n_lags, target, params):
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
    
    # One-hot encoding for categorical features
    province_encoded = pd.get_dummies(data['PROVINCE'], prefix='PROVINCE', drop_first=True, dtype=int)
    data = pd.concat([data, province_encoded], axis=1)
    
    estimation_encoded = pd.get_dummies(data['Estimation_Method/Méthode_destimation'], prefix='Estimation_Method', drop_first=True, dtype=int)
    data = pd.concat([data, estimation_encoded], axis=1)
    
    encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"])
    data = encoder.fit_transform(data, data[target])
    
    features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
               [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
               list(province_encoded.columns) + \
               ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"] + \
               list(estimation_encoded.columns)

    if 'Region' in data.columns:
        features.append('Region')

    train_data = data[data['Reporting_Year/Année'] < start_year]
    test_data = data[data['Reporting_Year/Année'] >= start_year]

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=42, **params))
        ])
        
        pipeline.fit(X_train, y_train)

        # Evaluate model performance on the test set
        y_pred = pipeline.predict(X_test)
        metrics = {
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'R² Score': r2_score(y_test, y_pred)
        }

        # Log metrics in MLflow
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log the trained model with input example and signature
        input_example = X_train.head(1)
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=input_example, signature=signature)

        # Save model locally
        model_directory = '/home/rutholasupo/2500_Labs/model'
        model_filename = 'random_forest_model.pkl'
        os.makedirs(model_directory, exist_ok=True)
        model_path = os.path.join(model_directory, model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        print(f"Model saved to {model_path}")
        print(f"Run ID: {run.info.run_id}")

        # Update the configuration with the latest run_id and parameters
        config['run_id'] = run.info.run_id
        config['start_year'] = start_year
        config['n_lags'] = n_lags
        config['target'] = target
        config['model_params'] = params

        # Save the updated configuration back to the YAML file
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)

    return pipeline, metrics

def main():
    # Prompt the user for input parameters
    start_year = input(f"Enter start year for training data (default: {config['start_year']}): ") or config['start_year']
    n_lags = input(f"Enter number of lags to create (default: {config['n_lags']}): ") or config['n_lags']
    target = input(f"Enter target variable for prediction (default: {config['target']}): ") or config['target']
    n_estimators = input(f"Enter number of trees in the forest (default: {config['model_params']['n_estimators']}): ") or config['model_params']['n_estimators']
    max_depth = input(f"Enter maximum depth of the tree (default: {config['model_params']['max_depth']}): ") or config['model_params']['max_depth']

    # Convert input parameters to the correct types
    start_year = int(start_year)
    n_lags = int(n_lags)
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }

    # Run the load_data.py script
    subprocess.run(["python3", "src/load_data.py"], check=True)

    # Run the preprocess.py script
    subprocess.run(["python3", "src/preprocess.py"], check=True)

    # Run the feature_engineering.py script
    subprocess.run(["python3", "src/feature_engineering.py"], check=True)

    train_path = "/home/rutholasupo/2500_Labs/data/processed/train_processed.csv"
    test_path = "/home/rutholasupo/2500_Labs/data/processed/test_processed.csv"
    combined_data_path = "/home/rutholasupo/2500_Labs/data/processed/combined_data.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    combined_df = combine_dataframe(df_train, df_test)
    combined_df.to_csv(combined_data_path, index=False)
    print(f"Combined data saved to {combined_data_path}")

    model, metrics = train_model(combined_df, start_year, n_lags, target, params)
    print("Model training complete and saved.")
    print("Metrics:\n", metrics)

if __name__ == "__main__":
    main()