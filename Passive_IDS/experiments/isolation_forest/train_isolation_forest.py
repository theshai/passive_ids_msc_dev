import pandas as pd
import joblib

import preprocessing.encoderer as enc
import preprocessing.splitter as splt
from preprocessing.dataset_config_info import DATASET_CONFIG
import preprocessing.analyzer as an
import preprocessing.cleaner as cl
import preprocessing.inspector as insp

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def main():

    dataset_name = "unsw_nb15_live_normal"

    # ---------------------------------------------------------
    # Load locally captured NORMAL traffic
    # ---------------------------------------------------------

    """
    df = pd.read_csv(
        "unsw_live_normal_capture.csv"
    )
    """
    config = DATASET_CONFIG[dataset_name]

    df , report = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    an.print_report(report)

    print("\nRows:", len(df))
    print("Columns:", len(df.columns))

    #clean the df, live might have doubles etc....
    df = cl.clean_dataset(df)
        
    #build new report
    report = insp.inspect_dataset(
            df=df,
            label_column="Label" #cleaner strip spaces
        )

    print("\nAfter cleaning:\n")
    an.print_report(report)

    input("\nfirst check")


    # ---------------------------------------------------------
    # All remaining columns come directly from unsw_extractor
    # ---------------------------------------------------------

    X = df.copy()

    print(
        "X shape before encoding:",
        X.shape
    )

    print(
        "X columns:",
        X.columns.tolist()
    )

    input("\nsecond check")

    # ---------------------------------------------------------
    # Split normal data
    #
    # There is no y because this is unsupervised.
    # ---------------------------------------------------------

    X_train, X_test = splt.split_dataset_unsupervised(
        X,
        dataset_name,
        test_size=0.20,
        random_state=42
    )


    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    input("\nthird check")


    # ---------------------------------------------------------
    # Use existing encoder
    # ---------------------------------------------------------

    encoder = enc.create_encoder(
        X,
        dataset_name
    )


    X_train_encoded = encoder.fit_transform(
        X_train
    )


    X_test_encoded = encoder.transform(
        X_test
    )


    print(
        "Encoded train:",
        X_train_encoded.shape
    )

    print(
        "Encoded test:",
        X_test_encoded.shape
    )

    print(
    "Encoded train fields:"
)

    print(
        X_train_encoded.columns.tolist()
    )


    input("\nforth check")

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------

    isolation_model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )


    isolation_model.fit(
        X_train_encoded
    )


    # ---------------------------------------------------------
    # Test against held-out NORMAL traffic
    # ---------------------------------------------------------

    predictions = isolation_model.predict(
        X_test_encoded
    )

    scores = isolation_model.decision_function(
        X_test_encoded
    )


    normal_count = (
        predictions == 1
    ).sum()

    anomaly_count = (
        predictions == -1
    ).sum()


    print("\n========================================")
    print("ISOLATION FOREST RESULTS")
    print("========================================")

    print(
        "Normal:",
        normal_count
    )

    print(
        "Anomaly:",
        anomaly_count
    )

    print(
        "Total:",
        len(predictions)
    )

    print(
        "Anomaly percentage:",
        (
            anomaly_count
            /
            len(predictions)
        ) * 100
    )

    input("\nfifth check")

    # ---------------------------------------------------------
    # Show most anomalous local-normal flows
    # ---------------------------------------------------------

    results = X_test.copy()

    results["prediction"] = predictions

    results["anomaly_score"] = scores


    print("\n========================================")
    print("MOST ANOMALOUS FLOWS")
    print("========================================")

    print(
        results
        .sort_values(
            "anomaly_score"
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    
    input("\nsixth check...going to save joblib....")

    # ---------------------------------------------------------
    # Save model + YOUR encoder
    # ---------------------------------------------------------

    joblib.dump(
        isolation_model,
        "models/unswnb15_isolation_forest/live_isolation_forest_v1.joblib"
    )


    joblib.dump(
        encoder,
        "models/unswnb15_isolation_forest/live_isolation_encoder_v1.joblib"
    )


if __name__ == "__main__":
    main()