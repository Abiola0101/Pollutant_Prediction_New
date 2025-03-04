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
 