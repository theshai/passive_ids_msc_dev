import numpy as np
import pandas as pd

#this version breaks when it is unsupervised
"""
def inspect_dataset(
    df: pd.DataFrame,
    label_column: str
) -> dict:

    numeric_df = df.select_dtypes(include=np.number)

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(
            np.isinf(numeric_df.to_numpy()).sum()
        ),
        "categorical_columns": df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist(),
        "label_counts": df[label_column].value_counts().to_dict(),
        "label_percentages": (
            df[label_column]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_dict()
        )
    }

    return report
"""

#-------------------------------------------------------
#This version supports both version of dataset
#-------------------------------------------------------

def inspect_dataset(
    df: pd.DataFrame,
    label_column: str = None
) -> dict:

    numeric_df = df.select_dtypes(
        include=np.number
    )

    report = {
        "rows": df.shape[0],

        "columns": df.shape[1],

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "infinite_values": int(
            np.isinf(
                numeric_df.to_numpy()
            ).sum()
        ),

        "categorical_columns":
            df.select_dtypes(
                include=[
                    "object",
                    "string"
                ]
            )
            .columns
            .tolist()
    }


    # ---------------------------------------------------------
    # Only analyze labels when a label column exists
    # ---------------------------------------------------------

    if (
        label_column is not None
        and label_column in df.columns
    ):

        report["label_counts"] = (
            df[label_column]
            .value_counts()
            .to_dict()
        )

        report["label_percentages"] = (
            df[label_column]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
            .to_dict()
        )

    else:

        report["label_counts"] = {}

        report["label_percentages"] = {}


    return report