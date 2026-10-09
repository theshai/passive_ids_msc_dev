import preprocessing.loader as ld
import preprocessing.inspector as insp

from preprocessing.dataset_config_info import DATASET_CONFIG

import preprocessing.cleaner as cl
import preprocessing.preperrer as prep
import preprocessing.encoderer as enc
import preprocessing.scaler as sc
import preprocessing.featureselector as fs
import preprocessing.analyzer as an
import models.hyperparameter_tuner as tuner
import sklearn as sk

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

from datetime import datetime

import joblib


"""
UNSW-NB15 experiment - XGBoost classifier


Comparison:


1. BASELINE MODEL

   - official UNSW training file
   - official UNSW testing file
   - all regular features
   - one-hot encoding
   - NO feature selection
   - NO hyperparameter tuning
   - default/basic XGBoost model


2. REGULAR DATASET

   - official UNSW training file
   - official UNSW testing file
   - full feature set
   - one-hot encoding
   - mutual information feature selection
   - XGBoost hyperparameter tuning
   - ROC-AUC used for hyperparameter scoring
   - train best model
   - evaluate on official test set


3. LIVE-COMPATIBLE DATASET

   - official UNSW training file
   - official UNSW testing file
   - manually selected live-compatible features
   - one-hot encoding
   - NO additional feature selection
   - XGBoost hyperparameter tuning
   - ROC-AUC used for hyperparameter scoring
   - train best model
   - evaluate on official test set


The live feature list is already a manual feature restriction,
therefore no mutual information feature selection is applied
to the live dataset.
"""


# ==============================================================
# XGBOOST HYPERPARAMETER TUNER
#
# Uses ONLY the training data supplied to it.
#
# The official UNSW-NB15 test dataset is NOT used here.
#
# scoring="roc_auc" because this gave a much better balance
# for the Random Forest experiment, especially reducing FPR.
# ==============================================================





def main() -> None:


    print(
        "\n**** START TIME ****",
        datetime.now().strftime("%H:%M:%S")
    )


    # selected dataset name from DATASET_CONFIG
    dataset_name = "unsw_nb15"


    config = DATASET_CONFIG[
        dataset_name
    ]


    #load official UNSW-NB15 TRAINING dataset
    df_train, report_train = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )


   

    print(
        "UNSW-NB15 TRAINING DATASET"
    )

   
    an.print_report(
        report_train
    )


   
    # Load official UNSW-NB15 TESTING dataset
    
    df_test, report_test = an.analyze_dataset(
        file_path=config["file_path_test"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )


   
    print(
        "UNSW-NB15 TESTING DATASET"
    )

   

    an.print_report(
        report_test
    )


    #------------------------------------------------------------
    # Prepare X / y
    #
    # Training comes ONLY from official training file.
    #
    # Testing comes ONLY from official testing file.
    #------------------------------------------------------------

    X_train, y_train = prep.prepare_xy(

        df_train,

        dataset_name
    )


    X_test, y_test = prep.prepare_xy(

        df_test,

        dataset_name
    )

  
    print(
        "OFFICIAL DATASET SHAPES"
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


   #experiment 1 - baseline XGBoost with all regular features, no feature selection, no tuning
   
    print(
        "EXPERIMENT 1"
    )

    print(
        "BASELINE XGBOOST"
    )

    print(
        "ALL REGULAR FEATURES"
    )

    print(
        "NO FEATURE SELECTION / NO TUNING"
    )

    
    
    regular_encoder = enc.create_encoder(
        X_train,
        dataset_name
    )
  
    # Fit encoder ONLY on training dataset


    X_train_encoded = (

        regular_encoder.fit_transform(

            X_train
        )
    )

    # Transform official testing dataset
  

    X_test_encoded = (

        regular_encoder.transform(

            X_test
        )
    )


    print(
        "\nRegular encoded training:",
        X_train_encoded.shape
    )

    print(
        "Regular encoded testing:",
        X_test_encoded.shape
    )


    # ----------------------------------------------------------
    # Print encoded columns if needed
    # ----------------------------------------------------------

    print(
        "\nEncoded regular feature count:",
        X_train_encoded.shape[1]
    )


    # Uncomment if needed
    #
    # print(
    #     X_train_encoded.columns.tolist()
    # )



    # Baseline XGBoost (for the report)
    baseline_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )


    print(
        "\nTraining baseline XGBoost..."
    )


    baseline_model.fit(
        X_train_encoded,
        y_train
    )


    # Baseline predictions

    y_pred_baseline = baseline_model.predict(
        X_test_encoded
    )

    y_pred_proba_baseline = (
        baseline_model.predict_proba(
            X_test_encoded
        )[:, 1]
    )

    # Baseline metrics
   
    baseline_metrics = an.calculate_metrics(
        y_test,
        y_pred_baseline,
        y_pred_proba_baseline
    )


    print(
        "BASELINE XGBOOST RESULTS"
    )

    print(
        "Accuracy, Precision, Recall, "
        "F1, MCC, ROC-AUC, FPR"
    )

    print(
        baseline_metrics
    )

    input(
        "\nPress Enter to continue to "
        "feature selection and tuning..."
    )


  
    
    # EXPERIMENT 2


   

    print(
        "EXPERIMENT 2"
    )

    print(
        "REGULAR DATASET"
    )

    print(
        "MUTUAL INFO FEATURE SELECTION + XGBOOST TUNING"
    )

   
    
    # Feature selection
    # ONLY applied to regular dataset.
    # Same feature-selection configuration used in the
    # Random Forest experiment (unswnb15_full_pipe_vs_live_sensor_RF_v1_56.py).
    # thelogic is that it is the same dataset, so the same feature-selection configuration should be used for both experiments.
 

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
        "REGULAR FEATURE SELECTION"
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


    
    # Tune XGBoost  
    # Only official TRAINING data is passed into the tuner.
    # Official test data remains untouched.
   

    print(
        "\n============================================================"
    )

    print(
        "XGBOOST HYPERPARAMETER TUNING"
    )

    print(
        "REGULAR DATASET"
    )

    print(
        "SCORING = ROC-AUC"
    )

   

    """
    (
        regular_best_model,

        regular_best_parameters,

        regular_best_cv_score

    ) = tuner.tune_xgboost(

        X_train_selected,

        y_train,

        scoring="roc_auc"
    )
    """
    #got the best parameters from the previous run and used them to train the model directly without tuning again
    regular_best_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )
    """
    # ==========================================================
    # Print tuning results
    # ==========================================================

    print(
        "\n------------------------------------------------------------"
    )

    print(
        "BEST XGBOOST - REGULAR DATASET"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        "\nBest parameters:"
    )


    for parameter, value in regular_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nBest CV ROC-AUC:"
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
   
    # Train best regular model
    # RandomizedSearchCV already refits the best estimator,
    # but fitting again makes it explicit that we train
    # using the complete official training dataset.
       

    print(
        "TRAINING BEST REGULAR XGBOOST"
    )
 

    regular_model = regular_best_model

    regular_model.fit(
        X_train_selected,
        y_train
    )


    print(
        "Regular XGBoost training completed."
    )


    y_pred_regular = regular_model.predict(

        X_test_selected
    )


    y_pred_proba_regular = (

        regular_model.predict_proba(

            X_test_selected

        )[:, 1]
    )


   
    regular_metrics = an.calculate_metrics(
        y_test,
        y_pred_regular,
        y_pred_proba_regular
    )


 
    print(
        "REGULAR DATASET RESULTS"
    )

    print(
        "FEATURE SELECTION + XGBOOST TUNING"
    )

    
    print(
        regular_metrics
    )

    input(
        "\nPress Enter to continue to ")


    # ==========================================================
    #
    # EXPERIMENT 3
    #
    # LIVE SENSOR-COMPATIBLE FEATURE SET
    #
    # NO ADDITIONAL FEATURE SELECTION
    #
    # XGBOOST HYPERPARAMETER TUNING
    #
    # ==========================================================

    print(
        "\n\n============================================================"
    )

    print(
        "EXPERIMENT 3"
    )

    print(
        "LIVE SENSOR FEATURE SET"
    )

    print(
        "NO ADDITIONAL FEATURE SELECTION"
    )

    print(
        "XGBOOST TUNING"
    )

    print(
        "============================================================"
    )


    # ==========================================================
    # Live feature list
    #
    # These are the features available/reproducible by
    # the live network sensor.
    #
    # This list IS the manual feature restriction.
    # ==========================================================

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

        # ------------------------------------------------------
        # TTL currently excluded
        # ------------------------------------------------------

        # "sttl",
        # "dttl",

        "sload",

        "dload",

        "sloss",

        "dloss",

        "sinpkt",

        "dinpkt",

        "sjit",

        "djit",

        # ------------------------------------------------------
        # TCP window values currently excluded
        # ------------------------------------------------------

        # "swin",
        # "dwin",

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
    # Select live-compatible fields
    #
    # NO feature-selection algorithm will be applied after this.
    # ==========================================================

    X_train_live = X_train[

        live_features

    ].copy()


    X_test_live = X_test[

        live_features

    ].copy()


    print(
        "\nLive raw training:",
        X_train_live.shape
    )


    print(
        "Live raw testing:",
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
    # Fit live encoder ONLY using official training dataset
    # ==========================================================

    X_train_live_encoded = (

        live_encoder.fit_transform(

            X_train_live
        )
    )


    # ==========================================================
    # Transform official testing dataset
    # ==========================================================

    X_test_live_encoded = (

        live_encoder.transform(

            X_test_live
        )
    )


    print(
        "\n------------------------------------------------------------"
    )

    print(
        "LIVE ENCODED DATA"
    )

    print(
        "------------------------------------------------------------"
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
    # IMPORTANT
    #
    # NO fs.apply_feature_selection() HERE.
    #
    # The live_features list already restricts the dataset
    # to fields that can be reproduced by the sensor.
    # ==========================================================


    # ==========================================================
    # Tune live XGBoost
    #
    # ROC-AUC optimization
    # ==========================================================

    print(
        "\n============================================================"
    )

    print(
        "XGBOOST HYPERPARAMETER TUNING"
    )

    print(
        "LIVE SENSOR DATASET"
    )

    print(
        "SCORING = ROC-AUC"
    )

    print(
        "============================================================"
    )

    """
    (
        live_best_model,

        live_best_parameters,

        live_best_cv_score

    ) = tune_xgboost(

        X_train_live_encoded,

        y_train,

        scoring="roc_auc"
    )
    """
    live_best_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )

 
   

    print(
        "BEST XGBOOST - LIVE SENSOR DATASET"
    )

    """


    print(
        "\nBest parameters:"
    )


    for parameter, value in live_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )


    print(
        "\nBest CV ROC-AUC:"
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

    """

   
    # Train best live XGBoost
  

   

    print(
        "TRAINING BEST LIVE XGBOOST"
    )

  

    live_model = live_best_model


    live_model.fit(
        X_train_live_encoded,
        y_train
    )


    print(
        "Live XGBoost training completed."
    )


 
    # Predict official testing dataset
   

    y_pred_live = live_model.predict(

        X_test_live_encoded
    )


    y_pred_proba_live = (

        live_model.predict_proba(

            X_test_live_encoded

        )[:, 1]
    )


   
    # Live metrics
   

    live_metrics = an.calculate_metrics(
        y_test,
        y_pred_live,
        y_pred_proba_live
    )


    print(
        "\n------------------------------------------------------------"
    )

    print(
        "LIVE SENSOR FEATURE RESULTS"
    )

    print(
        "NO FEATURE SELECTION + XGBOOST TUNING"
    )

    print(
        "------------------------------------------------------------"
    )


    print(
        live_metrics
    )


   

    print(
        "Accuracy, Precision, Recall, "
        "F1, MCC, ROC-AUC, FPR"
    )


    # ==========================================================
    # SAVE ARTIFACTS
    #
    # Leave this disabled until we are happy with the
    # official test results.
    # ==========================================================

    

    input(
        "\nIf you continue you are going to overwrite "
        "the tested XGBoost files..."
    )


    joblib.dump(

        live_model,

        "models/unswnb15/xgboost_live_unsw_official_80_20.joblib"
    )


    joblib.dump(

        live_encoder,

        "models/unswnb15/xgboost_live_encoder_unsw_official_80_20.joblib"
    )


    # There is NO algorithmic feature selection in the
    # live pipeline anymore.
    #
    # Therefore the expected model columns are simply
    # ALL columns produced by the live encoder.

    live_encoded_columns = (

        X_train_live_encoded.columns.tolist()
    )


    print(
        "Live encoded columns:"
    )

    print(
        live_encoded_columns
    )


    joblib.dump(

        live_encoded_columns,

        "models/unswnb15/"
        "xgboost_live_selected_columns_unsw_official_80_20.joblib"
    )

  


    print(
        "\n**** END TIME ****",
        datetime.now().strftime("%H:%M:%S")
    )


if __name__ == "__main__":

    main()