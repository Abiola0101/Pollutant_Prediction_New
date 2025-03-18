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
 
## To implement the project:
 
# Model Training Script
 This repository contains a script for training a machine learning model using a RandomForestRegressor.
 
## Setup Instructions
 
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
 
2. **Create a virtual environment**:
python3 -m venv venv
 
3. **Activate virual environment**:
For Mac/Linux: source venv/bin/activate
For Windows: .\venv\Scripts\activate
 
4. **Install the required packages**
Ensure you have a requirements.txt file in your repository. If not, create one with the necessary dependencies:
pandas
scikit-learn
category_encoders
numpy
pickle-mixin
mlflow
pyyaml
 
5. **Install dependencies**
pip install -r requirements.txt
 
6. **Run the training script**
python src/train.py --config configs/train_config.yaml
 
7. Viewing MLflow Metrics
   - **Start the MLflow tracking server**
      mlflow ui
   - **Open the MLflow UI: Open your web browser and navigate to http://localhost:5000 (or the specified port if different).**
   - **Open the MLflow UI: Open your web browser and navigate to http://localhost:5000 (or the specified port if different).**
 
8. Running Predictions
   - **Run the prediction script**
      python src/predict.py --config configs/train_config.yaml --combined_data_path data/processed/combined_data.csv --end_year 2023
 
 9. Running Predictions with the predict_api
   - **Run the predict_api script**
 
There are 4 endpoints in this API
- "/predictor_home": "Home page for the API"
- "/health_status": "Displays the health status of the API"
- "v1/predict": "This version of API is based on RandomForest model"
- "v2/predict": "This version of API is based on GradientBoost model"
 
To run the predict_api.py, on a terminal run:
- python predict_api.py --config configs/train_config.yaml --combined_data_path data/processed/combined_data.csv --end_year 2023
- curl -X POST "http://127.0.0.1:5050/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "start_year": 2023,
         "end_year": 2023,
         "n_lags": 2,
         "target": "Total_Release_Water",
         "data": [
             {
                 "Reporting_Year/Année": 2022,
                 "Population": 50000,
                 "Number_of_Employees": 100,
                 "Release_to_Air(Fugitive)": 120.5,
                 "Release_to_Air(Other_Non-Point)": 60.2,
                 "Release_to_Air(Road dust)": 24,
                 "Release_to_Air(Spills)": 110.2,
                 "Release_to_Air(Stack/Point)": 16.5,
                 "Release_to_Air(Storage/Handling)": 45.2,
                 "Releases_to_Land(Leaks)": 111.2,
                 "Releases_to_Land(Other)": 24,
                 "Releases_to_Land(Spills)": 36,
                 "Sum_of_release_to_all_media_(<1tonne)": 16.6,
                 "PROVINCE": "ON",
                 "Estimation_Method/Méthode destimation": "Calculated",
                 "City": "Toronto",
                 "Facility_Name/Installation": "ABC Plant",
                 "NAICS Title/Titre_Code_SCIAN": "Chemical Manufacturing",
                 "NAICS/Code_SCIAN": 325110,
                 "Company_Name/Dénomination sociale de l'entreprise": "XYZ Corp"
             }
         ]
     }'
 
 Note: The port number in the curl script must match the one defined in the predict_api.py file.
 