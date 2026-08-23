import loader as ld
import inspector as insp
from dataset_config_info import DATASET_CONFIG
import cleaner as cl
import preperrer as prep
import encoderer as enc
import scaler as sc
import sklearn as sk
import splitter as splt
from sklearn.metrics import (accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)                                              
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import featureselector as fs
import analyzer as an

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

    

    X_train, X_test, y_train, y_test = splt.split_dataset(
        X,
        y,
        dataset_name,
        test_size=0.2,
        random_state=42
    )
    
    #encoding the categorical features using one-hot encoding, and scaling the features using standard scaling. The encoded and scaled features will be used to train a logistic regression model, and the model will be evaluated using various metrics.
    encoder = enc.create_encoder(X,dataset_name)

    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test) 

    print(X_train_encoded.head())

    xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
    )

    # Apply feature selection -select the method you want
    X_train_selected, X_test_selected = fs.apply_feature_selection(
        X_train_encoded,
        X_test_encoded,
        y_train,
        method="variance",  # Options: "none", "variance", "correlation", "mutual_info", "select_k_best"
        threshold=0.01
    )

    
    print("trying scaled training the model.....")
    scaler = sc.create_scaler(X_train_selected,dataset_name)

    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)  

    # calling generic analyzing function
    an.analize_model(X_train_scaled, X_test_scaled, y_train, y_test)
   

    
    """
    # first try with the selected features, and then train a new logistic regression model using the selected features. The new model will be evaluated using the same metrics as before.
    lr_model = sk.linear_model.LogisticRegression(max_iter=2000,random_state=42)
    lr_model.fit(X_train_selected, y_train)
    print("Logistic Regression model trained.....")

    y_pred = lr_model.predict(X_test_selected)
    # for roc-auc, we need the predicted probabilities for the positive class
    y_pred_proba = lr_model.predict_proba(X_test_selected)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Matthews Correlation Coefficient:", mcc)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))
    """
if __name__ == "__main__":
    main()