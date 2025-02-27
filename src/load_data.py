import pandas as pd
import warnings

def load_data(filepath1, filepath2):
    try:
        # Load the first CSV file
        df1 = pd.read_csv(filepath1, encoding='latin1', low_memory=False)
        
        # Handle missing values (optional)
        #df1.dropna(subset=['Reporting_Year/Année', 'NPRI_ID/No_INRP'], inplace=True)

        # Load the second CSV file
        df2 = pd.read_csv(filepath2, encoding='latin1', low_memory=False)

        # Check for missing columns or values
        if 'Reporting_Year/Année' not in df1.columns or 'NPRI_ID/No_INRP' not in df1.columns:
            warnings.warn("Missing columns for merging in df1.")
        
        if 'Reporting_Year/Année' not in df2.columns or 'NPRI_ID/No_INRP' not in df2.columns:
            warnings.warn("Missing columns for merging in df2.")
        
        # Return both dataframes regardless of missing values
        return df1, df2

    except FileNotFoundError:
        print(f"Error: One or both of the files were not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: One or both of the files are empty.")
        return None, None
    except pd.errors.ParserError:
        print(f"Error: There was an issue parsing one or both of the files.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
