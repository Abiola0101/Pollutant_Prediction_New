#  Project Title
Production Deployment of NPRI Machine Learning Project
 
## Team members
 
Abi Afolabi
 
Abiola Bakare  
 
Ruth Olasupo
 
 
## Project Description:
This assignment is to showcase steps to deploy already developed ML model for NPRI pollution data into production.

1. Utilize GitHub as a central repository to manage code, track changes, and facilitate collaboration among team members.
2. Implement Data Version Control (DVC) to version datasets and machine learning models efficiently. DVC integrates with Git to handle large files, enabling reproducibility and easy sharing of data and model versions.
3. Use the Google API Console to configure remote storage solutions, such as Google Drive, ensuring secure and scalable storage for data files.
4. Employ MLflow to track experiments, record model parameters, and monitor performance metrics, facilitating continuous evaluation and improvement of the deployed model
5. Showcase steps to containerize pollution prediction app using Docker desktop. Containerization offers several key advantages for deploying machine learning models in that it allows for:

- **Reproducibility**: Ensures a consistent environment across development, testing, and production phases.
- **Portability**: Allows your application to run reliably on any platform that supports Docker.
- **Isolation**: Keeps dependencies and configurations separate from the underlying host system.
- **Scalability**: Simplifies the process of scaling your application in production environments.
- **Versioning**: Enables version control for the entire application environment, beyond just the code. 


## To implement the project:
 
### Setup Instructions
 
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
 
2. **Create a virtual environment**:
make init-cpu
 
3. **Activate virual environment**:
For Mac/Linux: source venv/bin/activate
For Windows: .\venv\Scripts\activate
 
4. **Install the required packages**

Ensure you have a requirements.txt file in your repository. If not, create one with the necessary dependencies:
- category_encoders==2.6.2
- joblib==1.4.2
- numpy==1.26.0
- pandas==2.2.3
- scikit-learn==1.3.1
- dvc==3.59.0
- PyYAML==6.0.2
- Cython==3.0.12
- Flask==3.1.0
- mlflow==2.20.2
- contourpy==1.3.1  # Updated to a compatible version
- kiwisolver==1.4.8 # Updated to a compatible version

5. **Upgrade pip and install dependencies**
- Upgrade pip to the latest version:
  ```bash
  pip install --upgrade pip
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt

6. Running Predictions

Run the prediction script python src/predict.py --config configs/predict_config.yaml --combined_data_path data/processed/combined_data.csv --end_year 2023
Running Predictions with the predict_api

Run the predict_api script
There are 4 endpoints in this API

"/predictor_home": "Home page for the API"
"/health_status": "Displays the health status of the API"
"v1/predict": "This version of API is based on RandomForest model"
"v2/predict": "This version of API is based on GradientBoost model"
To run the predict_api.py, on a terminal run:

python predict_api.py --config configs/train_config.yaml --combined_data_path data/processed/combined_data.csv --end_year 2023
curl -X POST "http://127.0.0.1:5050/v1/predict"
-H "Content-Type: application/json"
-d '{ "start_year": 2023, "end_year": 2023, "n_lags": 2, "target": "Total_Release_Water", "data": [ { "Reporting_Year/Année": 2022, "Population": 50000, "Number_of_Employees": 100, "Release_to_Air(Fugitive)": 120.5, "Release_to_Air(Other_Non-Point)": 60.2, "Release_to_Air(Road dust)": 24, "Release_to_Air(Spills)": 110.2, "Release_to_Air(Stack/Point)": 16.5, "Release_to_Air(Storage/Handling)": 45.2, "Releases_to_Land(Leaks)": 111.2, "Releases_to_Land(Other)": 24, "Releases_to_Land(Spills)": 36, "Sum_of_release_to_all_media_(<1tonne)": 16.6, "PROVINCE": "ON", "Estimation_Method/Méthode destimation": "Calculated", "City": "Toronto", "Facility_Name/Installation": "ABC Plant", "NAICS Title/Titre_Code_SCIAN": "Chemical Manufacturing", "NAICS/Code_SCIAN": 325110, "Company_Name/Dénomination sociale de l'entreprise": "XYZ Corp" } ] }'
Note: The port number in the curl script must match the one defined in the predict_api.py file.

7. **Build the Docker image**
- Create Dockerfile for ML application
-- Dockerfile.mlapp 

- Create Dockerfile for MLflow
-- Dockerfile.mlflow 

- Create Docker Compose file (.yml)
-- docker-compose.yml

The docker compose file is necessary for:

i)   **Service Orchestration**: Manages startup order (ensuring MLflow is running before your ML app)

ii)  **Networking**: Creates an internal network allowing containers to communicate

iii) **Configuration Management**: Centralizes environment variables and port mappings

iv)  **Volume Management**: Simplifies mounting directories for data persistence

 
8. Configure training script to connect to the MLflow tracking server
**Get the tracking URI from environment variable or use default**
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:9000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
 
9. Build and run contianers

**run the code below:**

docker-compose up --build

10. Publish Docker Images and view them on Docker desktop
- ML Application API: http://localhost:(your-api-port)
- MLflow UI: http://localhost:(your-mlflow-port)

11. To interact with our docker


Build and run your application with monitoring enabled:

```bash
docker-compose up --build
```

Access your monitoring tools:
* Grafana: http://localhost:pip 3000 (login with admin/password or which ever you have set)
* Prometheus: http://localhost:9090