import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import category_encoders as ce
import numpy as np
import pickle
import os
from mlflow.models.signature import infer_signature
import yaml
import mlflow

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        mlflow.sklearn.autolog()

    def combine_dataframe(self, df1, df2):
        combined_dataframe = pd.concat([df1, df2], axis=0)
        combined_dataframe.reset_index(drop=True, inplace=True)
        return combined_dataframe

    def create_lags_no_group(self, df, feature, n_lags):
        for i in range(1, n_lags + 1):
            df[f'{feature}_lag{i}'] = df[feature].shift(i)
        return df

    def train_model(self, data, start_year, n_lags, target, params):
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
        
        encoder = ce.TargetEncoder(cols=['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 
                                         'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"])
        data = encoder.fit_transform(data, data[target])
        
        features = [f'{pollutant}_lag{i}' for pollutant in pollutants for i in range(1, n_lags + 1)] + \
                   [f'{feature}_lag{i}' for feature in additional_features for i in range(1, n_lags + 1)] + \
                   list(province_encoded.columns) + ['City', 'Facility_Name/Installation', 'NAICS Title/Titre_Code_SCIAN', 
                                                     'NAICS/Code_SCIAN', "Company_Name/Dénomination_sociale_de_l'entreprise"] + \
                   list(estimation_encoded.columns)

        if 'Region' in data.columns:
            features.append('Region')

        train_data = data[data['Reporting_Year/Année'] < start_year]
        test_data = data[data['Reporting_Year/Année'] >= start_year]

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(random_state=42, **(params if params else {})))])
        pipeline.fit(X_train, y_train)

        # Evaluate model performance
        y_pred = pipeline.predict(X_test)
        metrics = {
            'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'R² Score': r2_score(y_test, y_pred)
        }

        # Model save path from config
        model_directory = self.config['model_directory']
        model_filename = self.config['model_filename']
        model_path = os.path.join(model_directory, model_filename)

        os.makedirs(model_directory, exist_ok=True)

        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        print(f"Model saved to {model_path}")

        with mlflow.start_run() as run:
            mlflow.log_params(params)

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

            # Log the trained model
            input_example = X_train.head(1)
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=input_example, signature=signature)

            # Save model locally
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)

            print(f"Model saved to {model_path}")
            print(f"Run ID: {run.info.run_id}")

            # Update config with run details
            self.config['run_id'] = run.info.run_id
            self.config['start_year'] = start_year
            self.config['n_lags'] = n_lags
            self.config['target'] = target
            self.config['model_params'] = params

            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f)

        return pipeline, metrics

    def main(self):
        # Load dataset paths from config
        train_path = self.config['train_path']
        test_path = self.config['test_path']
        combined_data_path = self.config['combined_data_path']

        # Load data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Combine and save data
        combined_df = self.combine_dataframe(df_train, df_test)
        combined_df.to_csv(combined_data_path, index=False)
        print(f"Combined data saved to {combined_data_path}")

        # Retrieve model parameters from config
        start_year = self.config['start_year']
        n_lags = self.config['n_lags']
        target = self.config['target']
        params = self.config['model_params']

        model, metrics = self.train_model(combined_df, start_year, n_lags, target, params)
        print("Model training complete and saved.")
        print("Metrics:\n", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')

    args = parser.parse_args()

    trainer = ModelTrainer(args.config)
    trainer.main()
