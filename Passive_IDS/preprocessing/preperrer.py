from dataset_config_info import DATASET_CONFIG

def prepare_xy(df,dataset_name):
    """
    Prepares the features (X) and labels (y) for the given dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        dataset_name (str): The name of the dataset to prepare.

    Returns:
        tuple: A tuple containing the features (X) and labels (y).
    """
    config = DATASET_CONFIG[dataset_name]
    label_column = config["label_column"]

    # Separate features (X) and labels (y)
    X = df.drop(columns=[label_column]).copy()  # Features

    if config["target_already_numeric"]:
        y = df[label_column].copy()  # Labels
    else:
        # Convert labels to binary (0 for normal, 1 for attack)
        normal_labels = config.get("normal_labels", [])
        y = df[label_column].apply(lambda x: 0 if x in normal_labels else 1).copy()  # Labels

    
    return X, y