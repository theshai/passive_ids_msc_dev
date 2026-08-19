import sklearn as sk

def create_scaler(X, dataset_name):
    """
    Creates a standard scaler for the dataset.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        dataset_name (str): The name of the dataset to scale.   

    Returns:
        StandardScaler: The created standard scaler.
    """
    scaler = sk.preprocessing.StandardScaler()
    return scaler   

    