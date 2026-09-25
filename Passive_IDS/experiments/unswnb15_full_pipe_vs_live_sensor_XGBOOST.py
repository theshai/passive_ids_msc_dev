import preprocessing.loader as ld
import preprocessing.inspector as insp
from preprocessing.dataset_config_info import DATASET_CONFIG
import preprocessing.cleaner as cl
import preprocessing.preperrer as prep
import preprocessing.encoderer as enc
import preprocessing.scaler as sc
import sklearn as sk
import preprocessing.splitter as splt
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
import preprocessing.featureselector as fs
import preprocessing.analyzer as an
from models.hyperparameter_tuner import tune_random_forest
from sklearn.ensemble import RandomForestClassifier
import joblib

"""
This file is an experiment file  that compare full pipline result vs live sensor selected result
trying to prove that we get a good tradeoff from hand picking the features that are easy to produce
on live data

main assumption for the test we have the fs and hyperparameter results for model, we will run both full dataset 
and live(sensor) features through hot encoder and see if we get acceptable results.

I am not going to use a real pipeline because I have the values - saves some time.

"""

def main() -> None:

    """load and analyze the UNSW-NB15
     dataset and the CIC-IDS2017 dataset,
      and print the reports for both datasets."""   
    
    dataset_name = "unsw_nb15"

    config = DATASET_CONFIG[dataset_name]

    df , report = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    an.print_report(report)
    
    #get clean X and y (and binary 0/1)
    X,y = prep.prepare_xy(df,dataset_name)

    #break data to train 80% and test 20%
    X_train, X_test, y_train, y_test = splt.split_dataset(
        X,
        y,
        dataset_name,
        test_size=0.2,
        random_state=42
    )

    #testing columns before drop and encode
    print("\nshape of X_train",X_train.shape)


    #----------------------------------------------------------
    #encoding the categorical features using one-hot encoding,
    #also removing catergorical columns
    #----------------------------------------------------------
    encoder = enc.create_encoder(X,dataset_name)

    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test) 

    print("\nshape of X_train_encoded",X_train_encoded.shape)
    #print(encoder.get_feature_names_out().tolist())

   
    # now try withthe list of the sensor
    print("trying with sensor list")
    #use the list as base all fileds before fs

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
    #"sttl", causing the model to depend on one feature only
    #"dttl", causing the model to depend on one feature only
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    #"swin", for experimenting
    #"dwin", for experimenting
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
    #take only the features that we use for the sensor
    X_live=X[live_features] 
    print(X_live.shape)

    X_train_live, X_test_live, y_train_live, y_test_live = train_test_split(
    X_live,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    categorical_columns = [
    "proto",
    "service",
    "state"
    ]
   
    encoder_live = sk.compose.ColumnTransformer(
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
    
    encoder_live.set_output(transform="pandas")
    
    X_train_live_encoded = encoder_live.fit_transform(X_train_live)
    X_test_live_encoded = encoder_live.transform(X_test_live)

    print("after encoding live:",X_train_live_encoded.shape)
    
    #use same fs....it is only a subset not a new dataset
    X_train_live_selected, X_test_live_selected, selected_columns, removed_columns, method = fs.apply_feature_selection(
            X_train_live_encoded,
            X_test_live_encoded,
            y_train_live,
            method="mutual_info",
            threshold=0.015,
            k=20
        )

    print(f"live pipe encoded selected after fs{X_train_live_selected.shape}")

    # ----------------------------------------------------------
    # Baseline XGBoost
    # ----------------------------------------------------------

    live_model_xgb = XGBClassifier(
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


    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------

    print("\nTraining UNSW XGBoost...")

    live_model_xgb.fit(
        X_train_live_selected,
        y_train_live
    )


    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------

    y_pred_live_xgb = live_model_xgb.predict(
        X_test_live_selected
    )

    y_pred_proba_live_xgb = (
        live_model_xgb.predict_proba(
            X_test_live_selected
        )[:, 1]
    )


    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    live_metrics_xgb = an.calculate_metrics(
        y_test_live,
        y_pred_live_xgb,
        y_pred_proba_live_xgb
    )


    print("\n--------------------------------------------")
    print("UNSW-NB15 XGBOOST RESULTS")
    print("--------------------------------------------")

    print(
        "Accuracy, Precision, Recall, F1, "
        "MCC, ROC-AUC, FPR"
    )

    print(live_metrics_xgb)

    print("saving trained nase model and live encoder.....")

    #------------------------------------------------------------------------
    #save all artifacts to be used later on raw data for predicition
    #------------------------------------------------------------------------

    joblib.dump(
     live_model_xgb,
    "models/unswnb15/xgboost_live_unsw.joblib"
    ) 

    joblib.dump(
     encoder_live,
    "models/unswnb15/xgboost_live_encoder_unsw.joblib"
    ) 

    print("selected columns:",selected_columns)
    joblib.dump(
     selected_columns,
    "models/unswnb15/xgboost_live_selected_columns_unsw.joblib"
    )  



if __name__ == "__main__":
    main()