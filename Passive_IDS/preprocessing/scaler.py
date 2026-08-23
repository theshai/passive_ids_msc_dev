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

    #concert back to pandas dataframe after scaling
    scaler.set_output(transform="pandas")
    
    return scaler   

    