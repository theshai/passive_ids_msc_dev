import numpy as np
import pandas as pd
from preprocessing.cleaner import clean_dataset


def test_clean_dataset(base_unsw):
    # Test the clean_dataset function
    cleaned = clean_dataset(base_unsw)
    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.isnull().sum().sum() == 0  # No missing values
    assert cleaned.duplicated().sum() == 0  # No duplicate rows
    assert cleaned.shape[0] <= base_unsw.shape[0]  # Rows should not increase
    assert cleaned.shape[1] == base_unsw.shape[1]  # Columns should remain the same
    assert cleaned.columns.equals(base_unsw.columns)  # Column names should remain the same
    assert cleaned.select_dtypes(include=[np.number]).apply(lambda x: x.isnull().sum()).sum() == 0  # No missing values in numeric columns
    