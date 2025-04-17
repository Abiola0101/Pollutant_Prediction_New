import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
from prometheus_client import start_http_server, Gauge
from logging_config import configure_logging
import os


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Get the root directory of the project (assumes this script runs from inside the project)
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(project_root, ".."))  # go one level up if needed

# Then inside your config parsing block
with open(args.config) as f:
    config = yaml.safe_load(f)

# Resolve paths
combined_data_path = os.path.join(project_root, config["combined_data_path"])
test_path = os.path.join(project_root, config["test_path"])
train_path = os.path.join(project_root, config["train_path"])






# Everything below this line is unchanged
# Configure logging
logger = configure_logging()
logger.debug("Logger configured successfully.")

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        mlflow.sklearn.autolog()
        logger.debug("ModelTrainer initialized with config path: %s", config_path)

        # Initialize Prometheus metrics
        self.rmse_gauge = Gauge('model_rmse', 'Root Mean Squared Error of the model')
        self.mae_gauge = Gauge('model_mae', 'Mean Absolute Error of the model')
        self.r2_gauge = Gauge('model_r2', 'R² Score of the model')

    def combine_dataframe(self, df1, df2):
        combined_dataframe = pd.concat([df1, df2], axis=0)
        combined_dataframe.reset_index(drop=True, inplace=True)
        return combined_dataframe

    def create_lags_no_group(self, df, feature, n_lags):
        for i in range(1, n_lags + 1):
            df[f'{feature}_lag{i}'] = df[feature].shift(i)
        return df

    def get_or_create_experiment(self, experiment_name, artifact_location=None):
        existing_exp = mlflow.get_experiment_by_name(experiment_name)
        if existing_exp is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            logger.info(f"Created experiment '{experiment_name}' with ID {experiment_id}")
        else:
            experiment_id = existing_exp.experiment_id
            logger.info(f"Experiment '{experiment_name}' already exists with ID {experiment_id} at {existing_exp.artifact_location}")

        mlflow.set_experiment(experiment_name)
        return experiment_id

    def train_model(self, data, start_year, n_lags, target, params):
        try:
            logger.info("Starting model training...")
            """
            # Set the MLflow tracking URI
            # mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            # mlflow.set_tracking_uri(mlflow_tracking_uri)

            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config["mlflow"]["tracking_uri"])
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            
            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            MLFLOW_PATH = os.path.join(PROJECT_ROOT, 'mlruns/')
            experiment_name = "Pollutant Prediction"
            artifact_uri = MLFLOW_PATH
            
            experiment_id = self.get_or_create_experiment(experiment_name, artifact_location=artifact_uri)
            logger.info(f"Using experiment_id: {experiment_id}")
            """
            # This is new
            

            # Safely get mlflow config
            mlflow_config = self.config.get("mlflow", {})
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_config.get("tracking_uri", "http://localhost:5000"))
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            MLFLOW_PATH = os.path.join(PROJECT_ROOT, 'mlruns/')
            artifact_uri = MLFLOW_PATH

            experiment_name = mlflow_config.get("experiment_name", "Pollutant Prediction")
            experiment_id = self.get_or_create_experiment(experiment_name, artifact_location=artifact_uri)
            logger.info(f"Using experiment_id: {experiment_id}")
            
            # Nothing  has changed here
            pollutants = [target]
            additional_features = [
                'Population', 'Number_of_Employees', 'Release_to_Air(Fugitive)', 'Release_to_Air(Other_Non-Point)',
                'Release_to_Air(Road dust)', 'Release_to_Air(Spills)', 'Release_to_Air(Stack/Point)',
                'Release_to_Air(Storage/Handling)', 'Releases_to_Land(Leaks)', 'Releases_to_Land(Other)',
                'Releases_to_Land(Spills)', 'Sum_of_release_to_all_media_(<1tonne)'
            ]
            
            # Preprocess data
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

            models = {
                'RandomForest': RandomForestRegressor(random_state=42, **(params if params else {})),
                'GradientBoosting': GradientBoostingRegressor(random_state=42, **(params if params else {}))
            }

            for model_name, model in models.items():
                logger.info(f"Training {model_name} model...")
                pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', model)])
                pipeline.fit(X_train, y_train)

                # Evaluate model performance
                y_pred = pipeline.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Expose metrics to Prometheus
                self.rmse_gauge.set(rmse)
                self.mae_gauge.set(mae)
                self.r2_gauge.set(r2)

                model_directory = os.path.join(os.getcwd(), self.config['model_directory'])  
                model_filename = f"{model_name}_{self.config['model_filename']}"
                model_path = os.path.join(model_directory, model_filename)

                os.makedirs(model_directory, exist_ok=True)

                with open(model_path, 'wb') as f:
                    pickle.dump(pipeline, f)

                logger.info(f"{model_name} model saved to {model_path}")

                # Log to MLflow
                with mlflow.start_run() as run:
                    mlflow.log_params(params)

                    pipeline.fit(X_train, y_train)

                    y_pred = pipeline.predict(X_test)
                    metrics = {
                        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
                        'R² Score': r2_score(y_test, y_pred)
                    }

                    for metric, value in metrics.items():
                        mlflow.log_metric(metric, value)

                    input_example = X_train.head(1)
                    signature = infer_signature(X_train, y_train)
                    mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=input_example, signature=signature)

                    with open(model_path, 'wb') as f:
                        pickle.dump(pipeline, f)

                    logger.info(f"{model_name} model saved to {model_path}")
                    logger.info(f"Run ID: {run.info.run_id}")

                    self.config[f'{model_name}_run_id'] = run.info.run_id

                with open(self.config_path, 'w') as f:
                    yaml.safe_dump(self.config, f)

            return models, metrics
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def main(self):
        try:
            # Start the Prometheus server on port 8002
            start_http_server(8002)
            logger.info("Prometheus server started on port 8002")

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
            logger.info(f"Combined data saved to {combined_data_path}")

            # Retrieve model parameters from config
            start_year = self.config['start_year']
            n_lags = self.config['n_lags']
            target = self.config['target']
            params = self.config['model_params']

            #print("Data columns:", combined_df.columns.tolist())

            models, metrics = self.train_model(combined_df, start_year, n_lags, target, params)
            logger.info("Model training complete and saved.")
            logger.info("Metrics:\n" + str(metrics))
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    
    args = parser.parse_args()

    # ✅ Add this to read your YAML config
    with open(args.config) as f:
        config = yaml.safe_load(f)


    
    # ✅ Now pass it to your Trainer class (if applicable)
    logger.debug("Starting ModelTrainer with config: %s", args.config)
    trainer = ModelTrainer(args.config)
    trainer.main()
