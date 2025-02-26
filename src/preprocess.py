import pandas as pd
import numpy as np

def split_data(df, year_column, split_year):
    """
    Splits the dataset into training and testing sets based on a specified year.
    """
    if year_column not in df.columns:
        print(f"Error: '{year_column}' column not found in DataFrame.")
        return None, None

    df_copy = df.copy()

    # Ensure the year column is numeric
    df_copy[year_column] = pd.to_numeric(df_copy[year_column], errors='coerce', downcast='integer')
    
    # Drop rows where the year is missing
    df_copy = df_copy.dropna(subset=[year_column])

    # Split into training and testing sets
    train_data = df_copy[df_copy[year_column] <= split_year]
    test_data = df_copy[df_copy[year_column] > split_year]

    return train_data, test_data

def preprocess_data(df):
    """
    Preprocess the dataset by:
    1. Dropping rows with missing values in non-release columns.
    2. Filling release-related columns with zeros.
    """
    df_copy = df.copy()

    # Columns where rows with missing values should be dropped
    non_release_columns_to_drop_rows = ['Number_of_Employees', 'Facility_Name/Installation', 'City', 'Latitude', 'Longitude']
    
    # Drop rows with missing values in specified non-release columns
    df_copy = df_copy.dropna(subset=non_release_columns_to_drop_rows)

    # Release-related columns to fill with zeros
    release_columns = [
        'Release_to_Air(Fugitive)',
        'Release_to_Air(Other_Non-Point)',
        'Release_to_Air(Road dust)',
        'Release_to_Air(Spills)',
        'Release_to_Air(Stack/Point)',
        'Release_to_Air(Storage/Handling)',
        'Releases_to_Land(Leaks)',
        'Releases_to_Land(Other)',
        'Releases_to_Land(Spills)',
        'Releases_ to_Water_Bodies(Direct Discharges)',
        'Releases_ to_Water_Bodies(Leaks)',
        'Releases_ to_Water_Bodies(Spills)',
        'Sum_of_release_to_all_media_(<1tonne)'
    ]
    
    # Fill missing values with zeros in release columns
    df_copy[release_columns] = df_copy[release_columns].fillna(0)

    return df_copy