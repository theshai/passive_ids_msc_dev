import pandas as pd
import pytest

from preprocessing.loader import function_load_dataset_with_header, function_load_dataset_without_header

def test_function_load_dataset_with_header(base_unsw, tmp_path):
    # Create a temporary CSV file with a header
    file_path = tmp_path / "unsw_test.csv"
    
    base_unsw.to_csv(file_path, index=False)

    # Load the dataset using the function
    df = function_load_dataset_with_header(
        file_path=file_path,
        header=0,
        sep=","
    )

    assert isinstance(df, pd.DataFrame)  # Ensure the output is a DataFrame
    assert df.equals(base_unsw)  # Ensure the loaded DataFrame matches the original
    assert df.shape == base_unsw.shape  # Ensure the shape matches
    assert list(df.columns) == list(base_unsw.columns)  # Ensure the columns match
    assert df.isnull().sum().sum() == base_unsw.isnull().sum().sum()  # Ensure missing values match
    assert df.duplicated().sum() == base_unsw.duplicated().sum()  # Ensure duplicate rows match
    assert df.loc[1, 'dur'] == base_unsw.loc[1, 'dur']  # Ensure specific values match
    assert df.loc[2, 'proto'] == base_unsw.loc[2, 'proto']  # Ensure specific values match