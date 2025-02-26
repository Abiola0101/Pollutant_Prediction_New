import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def merge_datasets(df1, df2, merge_columns):
    """
    Merges two DataFrames on specified columns.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        merge_columns (list): List of column names to merge on.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    return pd.merge(df1, df2, on=merge_columns)


def sum_release_columns(df, release_columns):
    """
    Sums the values in specified release columns and adds the total as a new column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the release columns.
        release_columns (list): List of column names to sum.

    Returns:
        pd.DataFrame: The DataFrame with the sum of release columns added.
    """
    df['Total_Release_Water'] = df[release_columns].sum(axis=1)
    df = df.drop(release_columns, axis=1)
    return df

def create_regions(df, n_clusters):
    """
    Creates regions based on the latitude and longitude of facilities.

    Parameters:
        df (pd.DataFrame): The DataFrame containing latitude and longitude columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'Region' column.
    """
    kmeans = KMeans(n_clusters=5, random_state=42).fit(df[['Latitude', 'Longitude']])
    df['Region'] = kmeans.labels_
    return df

def drop_columns(df):
    """
    Drops columns that are not needed for analysis.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with columns dropped.
    """
    columns_to_drop = [
        'Substance_Name_(English)/Nom_de_substance_(Anglais)',
        'Units/Unités']
    return df.drop(columns=columns_to_drop)