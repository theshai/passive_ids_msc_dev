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
UNSW-NB15 experiment -random forest classifier

Comparison:

1. Baseline model with all features, 
   -No feature selection,
   -No hyperparameter tuning

2. REGULAR DATASET
   - official UNSW training file
   - official UNSW testing file
   - full feature set
   - one-hot encoding
   - mutual information feature selection
   - Random Forest hyperparameter tuning
   - train best model
   - evaluate on official test set

3. LIVE-COMPATIBLE DATASET
   - official UNSW training file
   - official UNSW testing file
   - manually selected live-compatible features
   - one-hot encoding
   - NO additional feature selection
   - Random Forest hyperparameter tuning
   - train best model
   - evaluate on official test set

"""


def main() -> None:
   
    print("\n***start time****",datetime.now().strftime("%H:%M:%S"))
   
    #load dataset configuration for UNSW-NB15
    dataset_name = "unsw_nb15"

    config = DATASET_CONFIG[
        dataset_name
    ]


    #load official UNSW-NB15 TRAINING dataset tarining
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

 


    #load official UNSW-NB15 TESTING dataset
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



    # skipping cleaner for this datset, no duplicates or missing values in the official UNSW-NB15 dataset
    
    #prepare X and y for training and testing datasets

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



   

    print(
        "\n\n------------------------------------------------"
    )

    print(
        "EXPERIMENT 1"
    )

    print(
        "REGULAR DATASET"
    )
    
    print(
        "------------------------------------------------"
    )


    # ----------------------------------------------------------
    # Create encoder for full regular dataset + remove columns that are marked for removal in the config
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

    #test the fields and make sure that the removed columns are gone and the categorical fields are one-hot encoded
    #print(X_train_encoded.columns.tolist())

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

    print("get metrics with all the features, no feature selection, no tuning, just to see how the model performs with all the features and no tuning.")
    
    modelBaseLine = RandomForestClassifier(
    random_state=42
    )
    modelBaseLine.fit(X_train_encoded, y_train)
    # Class predictions
    y_pred_baseline = modelBaseLine.predict(
        X_test_encoded
    )

    # Probability of class 1 (ATTACK)
    y_pred_proba_baseline = modelBaseLine.predict_proba(
        X_test_encoded
    )[:, 1]

    baseLine_metrics = an.calculate_metrics(

        y_test,

        y_pred_baseline,

        y_pred_proba_baseline
    )
    print(
        "\nBefore preprocessing  metrics:"
    )

    print(
        baseLine_metrics
    )

    print(
        "Accuracy, Precision, Recall, "
        "F1, MCC, ROC-AUC, FPR"
    )

    input("\nPress Enter to continue to feature selection and tuning...")





    #Apply fs only on regular dataset, live dataset will not have fs applied, only manual feature restriction
    #Already got the bets FS parameters from previous experiments, so we can skip the FS tuning here and just apply the best parameters directly.
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


   
    # Hyperparameter tuning 
    # Only TRAINING data is passed to the tuner.

    
    print(
        "RANDOM FOREST HYPERPARAMETER TUNING"
    )

    print(
        "REGULAR DATASET"
    )

    
    #Skip this process, I ran it before and got the results below, so we can just use the best parameters directly without running the tuning again.

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
        "BEST RANDOM FOREST - REGULAR DATASET"
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
   
    # Train best regular Random Forest
    # on complete official training set
  

   

    print(
        "TRAINING BEST REGULAR RANDOM FOREST"
    )



    regular_model = regular_best_model


    regular_model.fit(
        X_train_selected,
        y_train
    )

    print(
        "Regular model training completed."
    )


    # Predict official test dataset
   
    y_pred_regular = regular_model.predict(

        X_test_selected
    )


    y_pred_proba_regular = (

        regular_model.predict_proba(

            X_test_selected

        )[:, 1]
    )

    # Regular dataset metrics
  

    regular_metrics = an.calculate_metrics(
        y_test,
        y_pred_regular,
        y_pred_proba_regular
    )


    print("\n\n------------------------------------------------")

    print(
        "REGULAR DATASET RESULTS"
    )

    print(
        "WITH FEATURE SELECTION + TUNING"
    )

    print("------------------------------------------------")


    print(
        regular_metrics
    )

    
    #----------------------------------------------------------
    # EXPERIMENT 3
    # LIVE SENSOR-COMPATIBLE FEATURE SET
    # NO FEATURE SELECTION
    # The list below IS the manual feature restriction.
    #----------------------------------------------------------

    print(
        "\n\n----------------------------------------------------------"
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
        "----------------------------------------------------------"
    )

    #list was copined on live sensor  from live monitoring system availability, and the features that are not available in the live sensor were commented out. The list below is the final list of features that are available in the live sensor and can be used for training and testing the model.
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


    #----------------------------------------------------------
    # Select live-compatible features
    # This is the manual feature restriction.
    # NO mutual_info / feature selector is used after this.
    #---------------------------------------------------------

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

    #dounle cjheck live columns and make sure they are the same as the manual feature restriction list above
    for column in X_train_live.columns:

        print(
            column
        )


  
    # Live categorical fields, added because we dont use the regular dataset config for the live dataset, since we are manually restricting the features to only those that are available in the live sensor. The categorical fields are the same as the regular dataset, but we need to specify them here for the live dataset.


    categorical_columns = [

        "proto",

        "service",

        "state"
    ]


    
    # Create live encoder - could use the regular encoder, but we are creating a new one here for clarity and to avoid any potential issues with the regular encoder being used for the live dataset.
    

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

    #use the encoder to fit and transform the live training data, and then transform the live testing data. The encoder is fit only on the training data, and then used to transform both the training and testing data.

    X_train_live_encoded = (

        live_encoder.fit_transform(

            X_train_live
        )
    )


    #transform the live testing data using the fitted encoder, and then print the shapes of the encoded training and testing data.

    X_test_live_encoded = (

        live_encoder.transform(

            X_test_live
        )
    )


    print(
        "\n----------------------------------------"
    )

    print(
        "LIVE ENCODED DATA"
    )

    print(
        "----------------------------------------"
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

    #bypassing hyper tuner due to time, if metric not good we wiil run it later, for now we will use the best parameters from the regular dataset tuning, since the live dataset is a subset of the regular dataset, and the best parameters should be similar.

   
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
        "\n----------------------------------------"
    )

    print(
        "TRAINING BEST LIVE RANDOM FOREST"
    )

    print(
        "----------------------------------------"
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
    #bypass for best model from tuning, since we are using the same parameters as the regular dataset tuning, and the live dataset is a subset of the regular dataset, so the best parameters should be similar.
    live_model = live_best_model


    live_model.fit(

        X_train_live_encoded,

        y_train
    )


    print(
        "Live model training completed."
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

    """
    print(
        f"Best CV score:             {regular_best_cv_score:.6f}"
    )
    """

    print(
        "\nBest parameters:"
    )

    """
    for parameter, value in regular_best_parameters.items():

        print(
            f"  {parameter}: {value}"
        )
    """

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

    """
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

    """
    print(
        "\nOfficial live metrics:"
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

    #------------------------------------------------------------------------
    #save all artifacts to be used later on raw data for predicition
    #------------------------------------------------------------------------
    """
    joblib.dump(
     live_model,
    "models/random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib"
    ) 

    joblib.dump(
     encoder_live,
    "models/live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
    ) 

    print("selected columns:",selected_columns)
    joblib.dump(
     selected_columns,
    "models/live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
    )  
    """

    print("\n***end time****",datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":

    main()