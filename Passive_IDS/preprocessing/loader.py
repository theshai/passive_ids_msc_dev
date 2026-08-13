import pandas as pd
from pandas import read_csv
import argparse
import sys

"""
Note, that due to obselete dataset, we will not use the old kdd dataset, but instead we will use the new kdd dataset. The new kdd dataset is a CSV file with a header row and a comma as the delimiter. The first column is the label column, and the rest of the columns are features. The label column contains the class labels for each instance in the dataset. The features are a mix of numeric and categorical values.
So no need to supply header sepetrately, as the new kdd dataset has a header row and a comma as the delimiter. The function_load_dataset_with_header function will be used to load the new kdd dataset, and the function_load_dataset_without_header function will be used to load other datasets that do not have a header row or use a different delimiter.
"""

def function_load_dataset_with_header(file_path, header=0, sep=","):
    """
    Load a dataset from a CSV file.

    Parameters:
    - file_path: str, path to the CSV file.
    - header: int or list of int, row number(s) to use as the column names.
    - sep: str, delimiter to use.

    Returns:
    - df: pandas DataFrame containing the loaded dataset.
    """
    try:
        df = read_csv(file_path, header=header, sep=sep)
        print(f"Dataset loaded successfully from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None

def function_load_dataset_without_header(file_path, header=0, sep=","):
    """
    Load a dataset from a CSV file.

    Parameters:
    - file_path: str, path to the CSV file.
    - header: int or list of int, row number(s) to use as the column names.
    - sep: str, delimiter to use.

    Returns:
    - df: pandas DataFrame containing the loaded dataset.
    """
    try:
        df = read_csv(file_path, header=header, sep=sep)
        print(f"Dataset with no header loaded successfully from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("file_path")
    parser.add_argument(
        "--header",
        action="store_true",
        help="Dataset contains a header row"
    )

    args = parser.parse_args()

    if args.header:
        df = function_load_dataset_with_header(
            args.file_path
        )
    else:
        df = function_load_dataset_without_header(
            args.file_path
        )

    print(df.head())