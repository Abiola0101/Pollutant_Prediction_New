# config.py

import os
"""
class Config:
    # Base config settings
    MODEL_V1_PATH = '/home/abiola/Pollutant_Prediction_New/model/RandomForest_random_forest_model.pkl'
    MODEL_V2_PATH = '/home/abiola/Pollutant_Prediction_New/model/GradientBoosting_random_forest_model.pkl'

    # Other configuration variables can go here
"""

from pathlib import Path

class Config:
    # Get the absolute path to the root of the repo (where config.py is located)
    BASE_DIR = Path(__file__).resolve().parent

    # Define model paths relative to BASE_DIR
    MODEL_V1_PATH = BASE_DIR / 'models' / 'RandomForest_random_forest_model.pkl'
    MODEL_V2_PATH = BASE_DIR / 'models' / 'GradientBoosting_random_forest_model.pkl'

