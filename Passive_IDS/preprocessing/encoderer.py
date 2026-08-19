import sklearn as sk
from dataset_config_info import DATASET_CONFIG


def create_encoder(X,dataset_name):
    """
    Creates a one-hot encoder for categorical features in the dataset.

    Args:
        X (pd.DataFrame): The input DataFrame containing the features.
        dataset_name (str): The name of the dataset to encode.

    Returns:
        ColumnTransformer: The created one-hot encoder.
    """
    config = DATASET_CONFIG[dataset_name]
    columns_to_encode = config.get("categorical_columns", [])
    columns_to_drop = config.get("drop_columns", [])    

    encoder = sk.compose.ColumnTransformer(
        transformers=[
            (
                "categorical",
                sk.preprocessing.OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                columns_to_encode
            ),(
                "drop",
                "drop",
                columns_to_drop
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    encoder.set_output(transform="pandas")  # Set output to pandas DataFrame
    return encoder