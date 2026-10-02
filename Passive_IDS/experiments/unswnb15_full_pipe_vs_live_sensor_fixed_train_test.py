import preprocessing.loader as ld
import preprocessing.inspector as insp

from preprocessing.dataset_config_info import DATASET_CONFIG

import preprocessing.cleaner as cl
import preprocessing.preperrer as prep
import preprocessing.encoderer as enc
import preprocessing.scaler as sc
import preprocessing.featureselector as fs
import preprocessing.analyzer as an

import sklearn as sk

from models.hyperparameter_tuner import tune_random_forest
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

"""
UNSW-NB15 experiment

Comparison:

1. REGULAR DATASET
   - official UNSW training file
   - official UNSW testing file
   - full feature set
   - one-hot encoding
   - mutual information feature selection
   - Random Forest hyperparameter tuning
   - train best model
   - evaluate on official test set

2. LIVE-COMPATIBLE DATASET
   - official UNSW training file
   - official UNSW testing file
   - manually selected live-compatible features
   - one-hot encoding
   - NO additional feature selection
   - Random Forest hyperparameter tuning
   - train best model
   - evaluate on official test set

No model files are saved in this experiment.
"""


def main() -> None:
   
    print("\n***start time****",datetime.now().strftime("%H:%M:%S"))
    # ==========================================================
    # Dataset configuration
    # ==========================================================

    dataset_name = "unsw_nb15"

    config = DATASET_CONFIG[
        dataset_name
    ]


    # ==========================================================
    # Load official UNSW-NB15 TRAINING dataset
    # ==========================================================

    df_train, report_train = an.analyze_dataset(

        file_path=config["file_path"],

        dataset_name=dataset_name,

        label_column=config["label_column"],

        columns=config["columns"]
    )

   

    print(
        "\n========================================"
    )

    print(
        "UNSW-NB15 TRAINING DATASET"
    )

    print(
        "========================================"
    )


    an.print_report(
        report_train
    )


    # ==========================================================
    # Load official UNSW-NB15 TESTING dataset
    # ==========================================================

    df_test, report_test = an.analyze_dataset(

        file_path=config["file_path_test"],

        dataset_name=dataset_name,

        label_column=config["label_column"],

        columns=config["columns"]
    )

    
    

    print(
        "\n========================================"
    )

    print(
        "UNSW-NB15 TESTING DATASET"
    )

    print(
        "========================================"
    )


    an.print_report(
        report_test
    )


    # ==========================================================
    # Prepare X / y
    #
    # Training data comes ONLY from official training file
    # Testing data comes ONLY from official testing file
    # ==========================================================

    X_train, y_train = prep.prepare_xy(

        df_train,

        dataset_name
    )


    X_test, y_test = prep.prepare_xy(

        df_test,

        dataset_name
    )

   
    print("second time:",datetime.now().strftime("%H:%M:%S"))


    print(
        "\n========================================"
    )

    print(
        "OFFICIAL DATASET SHAPES"
    )

    print(
        "========================================"
    )


    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    
    # ==========================================================
    #
    # EXPERIMENT 1
    #
    # REGULAR / FULL UNSW-NB15 DATASET
    #
    # WITH FEATURE SELECTION
    #
    # ==========================================================

    print(
        "\n\n========================================"
    )

    print(
        "EXPERIMENT 1"
    )

    print(
        "REGULAR DATASET"
    )

    print(
        "WITH MUTUAL INFO FEATURE SELECTION"
    )

    print(
        "========================================"
    )


    # ----------------------------------------------------------
    # Create encoder for full regular dataset
    # ----------------------------------------------------------

    regular_encoder = enc.create_encoder(

        X_train,

        dataset_name
    )


    # ----------------------------------------------------------
    # Fit encoder ONLY using training data
    # ----------------------------------------------------------

    X_train_encoded = (
        regular_encoder.fit_transform(
            X_train
        )
    )


    # ----------------------------------------------------------
    # Transform official testing data
    # ----------------------------------------------------------

    X_test_encoded = (
        regular_encoder.transform(
            X_test
        )
    )


    print(
        "\nRegular encoded train:",
        X_train_encoded.shape
    )

    print(
        "Regular encoded test:",
        X_test_encoded.shape
    )


    # ==========================================================
    # FEATURE SELECTION
    #
    # ONLY APPLIED TO REGULAR DATASET
    # ==========================================================

    (
        X_train_selected,

        X_test_selected,

        selected_columns,

        removed_columns,

        method

    ) = fs.apply_feature_selection(

        X_train_encoded,

        X_test_encoded,

        y_train,

        method="mutual_info",

        threshold=0.015,

        k=20
    )


    print(
        "\n========================================"
    )

    print(
        "REGULAR FEATURE SELECTION"
    )

    print(
        "========================================"
    )


    print(
        "Method:",
        method
    )

    print(
        "Training before FS:",
        X_train_encoded.shape
    )

    print(
        "Training after FS:",
        X_train_selected.shape
    )

    print(
        "Testing after FS:",
        X_test_selected.shape
    )

    print(
        "Selected feature count:",
        len(selected_columns)
    )


    print(
        "\nSELECTED REGULAR FEATURES"
    )


    for column in selected_columns:

        print(
            column
        )


    print(
        "\nRemoved feature count:",
        len(removed_columns)
    )


    # ==========================================================
    # Hyperparameter tuning
    # REGULAR DATASET
    #
    # IMPORTANT:
    # Only TRAINING data is passed to the tuner.
    # ==========================================================

    print(
        "\n========================================"
    )

    print(
        "RANDOM FOREST HYPERPARAMETER TUNING"
    )

    print(
        "REGULAR DATASET"
    )

    print(
        "========================================"
    )

    """
    (
        regular_best_model,

        regular_best_parameters,

        regular_best_cv_score

    ) = tune_random_forest(

        X_train_selected,

        y_train
    )
    """
    regular_best_model=RandomForestClassifier(  class_weight='balanced',
                                                max_depth=40,
                                                min_samples_leaf=2,
                                                min_samples_split=10,
                                                n_estimators=200,
                                                n_jobs=1,
                                                random_state=42
                                                )



    print(
        "\n----------------------------------------"
    )

    print(
        "BEST RANDOM FOREST - REGULAR DATASET"
    )

    print(
        "----------------------------------------"
    )

    """
    print(
        "\nBest parameters:"
    )


    for parameter, value in regular_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nBest cross-validation score:"
    )

    print(
        f"  {regular_best_cv_score:.6f}"
    )


    print(
        "\nBest estimator:"
    )

    print(
        regular_best_model
    )

    """
    # ==========================================================
    # Train best regular Random Forest
    # on complete official training set
    # ==========================================================

    print(
        "\n========================================"
    )

    print(
        "TRAINING BEST REGULAR RANDOM FOREST"
    )

    print(
        "========================================"
    )


    regular_model = regular_best_model


    regular_model.fit(

        X_train_selected,

        y_train
    )


    print(
        "Regular model training completed."
    )


    # ==========================================================
    # Predict official test dataset
    # ==========================================================

    y_pred_regular = regular_model.predict(

        X_test_selected
    )


    y_pred_proba_regular = (

        regular_model.predict_proba(

            X_test_selected

        )[:, 1]
    )


    # ==========================================================
    # Regular dataset metrics
    # ==========================================================

    regular_metrics = an.calculate_metrics(

        y_test,

        y_pred_regular,

        y_pred_proba_regular
    )


    print(
        "\n========================================"
    )

    print(
        "REGULAR DATASET RESULTS"
    )

    print(
        "WITH FEATURE SELECTION + TUNING"
    )

    print(
        "========================================"
    )


    print(
        regular_metrics
    )

    
    # ==========================================================
    #
    # EXPERIMENT 2
    #
    # LIVE SENSOR-COMPATIBLE FEATURE SET
    #
    # NO FEATURE SELECTION
    #
    # The list below IS the manual feature restriction.
    #
    # ==========================================================

    print(
        "\n\n========================================"
    )

    print(
        "EXPERIMENT 2"
    )

    print(
        "LIVE SENSOR FEATURE SET"
    )

    print(
        "NO ADDITIONAL FEATURE SELECTION"
    )

    print(
        "========================================"
    )


    live_features = [

        "proto",

        "service",

        "state",

        "dur",

        "spkts",

        "dpkts",

        "sbytes",

        "dbytes",

        "rate",

        # Not used by live sensor model -added for test
        #"sttl",
        #"dttl",

        "sload",

        "dload",

        "sloss",

        "dloss",

        "sinpkt",

        "dinpkt",

        "sjit",

        "djit",

        # Not used by live sensor model
       #"swin",
       #"dwin",

        "stcpb",

        "dtcpb",

        "tcprtt",

        "synack",

        "ackdat",

        "smean",

        "dmean",

        "response_body_len",

        "is_sm_ips_ports"
    ]


    # ==========================================================
    # Select live-compatible features
    #
    # This is the manual feature restriction.
    #
    # NO mutual_info / feature selector is used after this.
    # ==========================================================

    X_train_live = X_train[
        live_features
    ].copy()


    X_test_live = X_test[
        live_features
    ].copy()


    print(
        "\nLive raw training shape:",
        X_train_live.shape
    )

    print(
        "Live raw testing shape:",
        X_test_live.shape
    )


    print(
        "\nLIVE RAW FEATURES"
    )


    for column in X_train_live.columns:

        print(
            column
        )


    # ==========================================================
    # Live categorical fields
    # ==========================================================

    categorical_columns = [

        "proto",

        "service",

        "state"
    ]


    # ==========================================================
    # Create live encoder
    # ==========================================================

    live_encoder = sk.compose.ColumnTransformer(

        transformers=[

            (

                "categorical",

                sk.preprocessing.OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False
                ),

                categorical_columns
            )
        ],

        remainder="passthrough"
    )


    live_encoder.set_output(
        transform="pandas"
    )


    # ==========================================================
    # Fit live encoder ONLY using official training data
    # ==========================================================

    X_train_live_encoded = (

        live_encoder.fit_transform(

            X_train_live
        )
    )


    # ==========================================================
    # Transform official testing data
    # ==========================================================

    X_test_live_encoded = (

        live_encoder.transform(

            X_test_live
        )
    )


    print(
        "\n========================================"
    )

    print(
        "LIVE ENCODED DATA"
    )

    print(
        "========================================"
    )


    print(
        "Live encoded training:",
        X_train_live_encoded.shape
    )

    print(
        "Live encoded testing:",
        X_test_live_encoded.shape
    )


    print(
        "Live encoded feature count:",
        X_train_live_encoded.shape[1]
    )


    # ==========================================================
    # Hyperparameter tuning
    # LIVE SENSOR DATASET
    #
    # NO feature selection here.
    #
    # Tune the Random Forest directly on the encoded
    # live-compatible training features.
    # ==========================================================

    print(
        "\n========================================"
    )

    print(
        "RANDOM FOREST HYPERPARAMETER TUNING"
    )

    print(
        "LIVE SENSOR DATASET"
    )

    print(
        "========================================"
    )

    '''
    (
        live_best_model,

        live_best_parameters,

        live_best_cv_score

    ) = tune_random_forest(

        X_train_live_encoded,

        y_train
    )
    '''

    live_best_model=RandomForestClassifier(class_weight='balanced',
                                           max_depth=40,
                                           min_samples_leaf=2,
                                           min_samples_split=10,
                                           n_estimators=200,
                                           n_jobs=1,
                                           random_state=42
                                           )

    """
    print(
        "\n----------------------------------------"
    )

    print(
        "BEST RANDOM FOREST - LIVE SENSOR DATASET"
    )

    print(
        "----------------------------------------"
    )


    print(
        "\nBest parameters:"
    )


    for parameter, value in live_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nBest cross-validation score:"
    )

    print(
        f"  {live_best_cv_score:.6f}"
    )


    print(
        "\nBest estimator:"
    )

    print(
        live_best_model
    )


    # ==========================================================
    # Train best live Random Forest
    # on complete official training set
    # ==========================================================

    print(
        "\n========================================"
    )

    print(
        "TRAINING BEST LIVE RANDOM FOREST"
    )

    print(
        "========================================"
    )
    
    live_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='log2',
    class_weight=None,
    n_jobs=1,
    random_state=42
    )
    """

    live_model = live_best_model


    live_model.fit(

        X_train_live_encoded,

        y_train
    )


    print(
        "Live model training completed."
    )


    # ==========================================================
    # Predict official testing dataset
    # ==========================================================

    y_pred_live = live_model.predict(

        X_test_live_encoded
    )


    y_pred_proba_live = (

        live_model.predict_proba(

            X_test_live_encoded

        )[:, 1]
    )


    # ==========================================================
    # Live metrics
    # ==========================================================

    live_metrics = an.calculate_metrics(

        y_test,

        y_pred_live,

        y_pred_proba_live
    )


    print(
        "\n========================================"
    )

    print(
        "LIVE SENSOR FEATURE RESULTS"
    )

    print(
        "NO FEATURE SELECTION + TUNING"
    )

    print(
        "========================================"
    )


    print(
        live_metrics
    )
    
    input("stop here.....")

    # ==========================================================
    #
    # FINAL COMPARISON
    #
    # ==========================================================

    print(
        "\n\n============================================================"
    )

    print(
        "FINAL RANDOM FOREST COMPARISON"
    )

    print(
        "============================================================"
    )


    # ----------------------------------------------------------
    # Regular
    # ----------------------------------------------------------

    print(
        "\nREGULAR DATASET"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        f"Raw features:              {X_train.shape[1]}"
    )

    print(
        f"Encoded features:          {X_train_encoded.shape[1]}"
    )

    print(
        "Feature selection:         mutual_info"
    )

    print(
        "MI threshold:              0.015"
    )

    print(
        f"Selected features:         {X_train_selected.shape[1]}"
    )

    print(
        f"Best CV score:             {regular_best_cv_score:.6f}"
    )


    print(
        "\nBest parameters:"
    )


    for parameter, value in regular_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nOfficial test metrics:"
    )

    print(
        regular_metrics
    )


    # ----------------------------------------------------------
    # Live
    # ----------------------------------------------------------

    print(
        "\nLIVE SENSOR DATASET"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        f"Raw live features:         {X_train_live.shape[1]}"
    )

    print(
        f"Encoded live features:     {X_train_live_encoded.shape[1]}"
    )

    print(
        "Feature selection:         NONE"
    )

    print(
        f"Best CV score:             {live_best_cv_score:.6f}"
    )


    print(
        "\nBest parameters:"
    )


    for parameter, value in live_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nOfficial test metrics:"
    )

    print(
        live_metrics
    )


    # ----------------------------------------------------------
    # Metric order
    # ----------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "METRIC ORDER"
    )

    print(
        "============================================================"
    )

    print(
        "Accuracy, Precision, Recall, "
        "F1, MCC, ROC-AUC, FPR"
    )

    print("\n***end time****",datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":

    main()