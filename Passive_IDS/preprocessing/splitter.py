import sklearn as sk
from preprocessing.dataset_config_info import DATASET_CONFIG

def split_dataset(X, y, dataset_name, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        y (pd.Series): The labels corresponding to the features.
        dataset_name (str): The name of the dataset to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    config = DATASET_CONFIG[dataset_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test