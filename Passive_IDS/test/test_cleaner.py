import numpy as np
import pandas as pd
from preprocessing.cleaner import clean_dataset


def test_clean_dataset(base_unsw):
    """
    Test the clean_dataset function.
    Ref base_unsw fixture in conftest.py
    """
    #make sure nan exist before cleaning
    assert base_unsw.isna().sum().sum() > 0  # Ensure there are missing values before cleaning
    # Test the clean_dataset function
    cleaned = clean_dataset(base_unsw)
    assert isinstance(cleaned, pd.DataFrame)  # Ensure the output is a DataFrame
    assert cleaned.isna().sum().sum() == 0  # Ensure there are no missing values after cleaning
    assert cleaned.loc[2, 'dur'] == base_unsw['dur'].median()  # Ensure missing values are filled with median
    assert cleaned.shape[0] > 0  # Ensure the cleaned DataFrame is not empty
    assert cleaned.isnull().sum().sum() == 0  # No missing values
    assert cleaned.duplicated().sum() == 0  # No duplicate rows
    assert cleaned.shape[0] <= base_unsw.shape[0]  # Rows should not increase
    assert cleaned.shape[1] == base_unsw.shape[1]  # Columns should remain the same
    assert cleaned.columns.equals(base_unsw.columns)  # Column names should remain the same
    assert cleaned.select_dtypes(include=[np.number]).apply(lambda x: x.isnull().sum()).sum() == 0  # No missing values in numeric columns
    