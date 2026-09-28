import sklearn as sk
import preprocessing.loader as ld
import preprocessing.inspector as insp
from sklearn.metrics import (accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        confusion_matrix,
        classification_report
)     
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def analize_model_random_forest(X_train, X_test, y_train, y_test):

        random_forset_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
        )

        random_forset_model.fit(X_train,y_train) 
        print("Random forest model trained.....")
        y_pred = random_forset_model.predict(X_test)

        y_pred_proba = random_forset_model.predict_proba( X_test)[:, 1]
        
        return calculate_metrics(y_test, y_pred, y_pred_proba)   
    


def analize_model_logostic_regression(X_train_scaled, X_test_scaled, y_train, y_test):                         

        # first model
        lr_model = sk.linear_model.LogisticRegression(max_iter=1000,random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        print("Logistic Regression model trained.....")

        y_pred = lr_model.predict(X_test_scaled)
        # for roc-auc, we need the predicted probabilities for the positive class
        y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

        """
        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))
        """
        return calculate_metrics(y_test, y_pred, y_pred_proba)

def analize_model_xgboost(X_train, X_test, y_train, y_test):   
        
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
        
        xgb_model.fit(X_train, y_train)
        print("XGBoost model trained.....")

        y_pred = xgb_model.predict(X_test)
        # for roc-auc, we need the predicted probabilities for the positive class
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
        return calculate_metrics(y_test, y_pred, y_pred_proba)

# standard function to calculate metrics (for all models)
def calculate_metrics(
    y_test,
    y_pred,
    y_pred_proba
):
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_pred_proba
    )

    tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
    ).ravel()

    fpr = fp / (fp + tn)

    return (
        accuracy,
        precision,
        recall,
        f1,
        mcc,
        roc_auc,
        fpr
    ) 

    # Basic analyzing of dataset

def analyze_dataset(
        file_path: str,
        dataset_name: str,
        label_column: str  | None = None,
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

    #report for basic analyzing
def print_report(
    report: dict
) -> None:

    print("\nDataset report")
    print("-" * 50)

    print(
        "Rows:",
        report["rows"]
    )

    print(
        "Columns:",
        report["columns"]
    )

    print(
        "Missing values:",
        report["missing_values"]
    )

    print(
        "Infinite values:",
        report["infinite_values"]
    )

    print(
        "Duplicate rows:",
        report["duplicate_rows"]
    )

    print("\nCategorical columns:")

    print(
        report["categorical_columns"]
    )


    # ------------------------------------------------------
    # Only print label information if labels exist
    # ------------------------------------------------------

    if report.get("label_counts"):

        print("\nLabel counts:")

        for label, count in (
            report["label_counts"].items()
        ):

            print(
                f"{label}: {count}"
            )


        print("\nLabel percentages:")

        for label, percentage in (
            report["label_percentages"].items()
        ):

            print(
                f"{label}: {percentage}%"
            )

def print_report_delete(report: dict) -> None:
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