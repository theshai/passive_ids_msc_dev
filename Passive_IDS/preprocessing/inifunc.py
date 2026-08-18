import loader as ld
import inspector as insp
from dataset_config_info import DATASET_CONFIG
import cleaner as cl
import preperrer as prep
import encoderer as enc
from sklearn.model_selection import train_test_split

"""generic function to analyze a dataset, given the file path, 
   ataset name, label column, and optional columns to load. 
   The function will load the dataset using the loader module,
   and then inspect the dataset using the inspector module. 
   The function will return the loaded DataFrame and the inspection report."""

def analyze_dataset(
    file_path: str,
    dataset_name: str,
    label_column: str,
    columns: list[str] | None = None
):
    df = ld.function_load_dataset_with_header(
        file_path=file_path,
        header=0,
        sep=","
    )

    report = insp.inspect_dataset(
        df=df,
        label_column=label_column
    )

    return df, report

def print_report(report: dict) -> None:
    print("\nDataset report")
    print("-" * 50)

    print("Rows:", report["rows"])
    print("Columns:", report["columns"])
    print("Missing values:", report["missing_values"])
    print("Infinite values:", report["infinite_values"])
    print("Duplicate rows:", report["duplicate_rows"])

    print("\nCategorical columns:")
    print(report["categorical_columns"])

    print("\nLabel counts:")
    for label, count in report["label_counts"].items():
        print(f"{label}: {count}")

    print("\nLabel percentages:")
    for label, percentage in report["label_percentages"].items():
        print(f"{label}: {percentage}%")

"need to dynamic dataset path, so that we can use different datasets for testing and training. The dataset path should be passed as a command line argument, and the default value should be the path to the UNSW-NB15 dataset."
def main() -> None:

    """load and analyze the UNSW-NB15
     dataset and the CIC-IDS2017 dataset,
      and print the reports for both datasets."""   
    
    dataset_name = "unsw_nb15"

    config = DATASET_CONFIG[dataset_name]

    df , report = analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    print_report(report)

    X,y = prep.prepare_xy(df,dataset_name)

    print("\nFeatures (X):")
    print(X.head())  

    encoder = enc.create_encoder(X,dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
     X,
     y,
     test_size=0.2,
     random_state=42,
     stratify=y
    )  

    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test) 
    
    """
    # Get encoded column names
    encoded_column_names = encoder.get_feature_names_out()

    print("\nEncoded columns:")
    for column in encoded_column_names:
        print(column)
    """

    dataset_name = "cic2017"

    config = DATASET_CONFIG[dataset_name]

    df , report = analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    print_report(report)

    "try cleaning the dataset using the cleaner module, and then inspect the cleaned dataset using the inspector module. The function will return the cleaned DataFrame and the inspection report."
     
    df_cleaned = cl.clean_dataset(df)
    
    report = insp.inspect_dataset(
        df=df_cleaned,
        label_column="Label"
    )
    print_report(report)

    X,y = prep.prepare_xy(df,dataset_name)

    print("\nFeatures (X):")
    print(X.head())  

    print("\nFeatures (X):")
    print(X.head())




"""
    file_path = (
        "datasets/RAW/UNSW-NB15/UNSW_NB15_training-set.csv"
    )

    # Step 1: Load the raw file.
    df = ld.function_load_dataset_with_header(
        file_path=file_path,
        header=0,
        sep=","
    )

    # The returned DataFrame is now passed to the inspector.
    report = insp.inspect_dataset(
        df=df,
        label_column="label"
    )

    print(df.head())

    print_report(report)
"""
if __name__ == "__main__":
    main()