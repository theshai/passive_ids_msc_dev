import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
   
    """
    Clean the dataset by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    "work on copy only, so that we don't modify the original dataframe"

    cleaned_df = df.copy()

    cleaned_df.columns = cleaned_df.columns.str.strip()  # Strip whitespace from column names
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    """
    # Handle missing values (example: fill with median)
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        median_value = cleaned_df[column].median()
        cleaned_df[column].fillna(median_value, inplace=True)"""

    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
    cleaned_df.dropna(inplace=True)  # Drop rows with any NaN values
    cleaned_df.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
    

    return cleaned_df