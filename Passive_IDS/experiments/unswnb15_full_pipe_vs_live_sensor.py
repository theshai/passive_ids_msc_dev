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

    #---------------------------------------------------------
    #Feature selection code, gets the best fs for the dataset
    #---------------------------------------------------------

    #first the options...
    #experimanting with different feature selection methods and thresholds, and analyzing the results using the generic 
    #analyzing function. The feature selection methods include variance thresholding, correlation thresholding, 
    #mutual information, and select k best. The thresholds for each method can be adjusted to see how they affect the model performance.
    """
    feature_selection_results = []
    feature_selection_results.append(["testing-model","method","parameters","features","accuracy","precision","recall","f1","mcc","roc_auc","fpr"])
    feature_selection_methods = [
    {"method": "none"}
    ]

    for threshold in [
        0,
        0.001,
        0.005,
        0.01
    ]:
        feature_selection_methods.append({
            "method": "variance",
            "threshold": threshold
        })

    for threshold in [
        0.80,
        0.90,
        0.95,
        0.99
    ]:
        feature_selection_methods.append({
            "method": "correlation",
            "threshold": threshold
        })

    for threshold in [
         0.005,
         0.0075,
         0.01,
         0.0125,
         0.015,
         0.02
        ]:
        feature_selection_methods.append({
            "method": "mutual_info",
            "threshold": threshold
        })

    for k in [
        20,
        40,
        60,
        80,
        100
    ]:
        feature_selection_methods.append({
            "method": "select_k_best",
            "k": k
        })

     #this part to get the best fs ....after you find it, dont call again unless you have a new model or a new ds
    
    for method_config in feature_selection_methods:
        x = method_config["method"]
        threshold = method_config.get("threshold", 0.0)
        k = method_config.get("k", 20)

        print(f"\nApplying feature selection method: {x} with threshold: {threshold} and k: {k}")
        
        X_train_selected, X_test_selected, selected_columns, removed_columns, method = fs.apply_feature_selection(
            X_train_encoded,
            X_test_encoded,
            y_train,
            method=x,
            threshold=threshold,
            k=k
        )

        # calling xgboost function
        accuracy, precision, recall, f1, mcc, roc_auc,fpr = an.analize_model_xgboost(X_train_selected, X_test_selected, y_train, y_test)
        feature_selection_results.append(["XGBoost",x, threshold if x != "select_k_best" else k, X_train_selected.shape[1], accuracy, precision, recall, f1, mcc, roc_auc,fpr])
       
        # calling random forest function
        accuracy, precision, recall, f1, mcc, roc_auc,fpr = an.analize_model_random_forest(X_train_selected, X_test_selected, y_train, y_test)
        feature_selection_results.append(["RandomForset",x, threshold if x != "select_k_best" else k, X_train_selected.shape[1], accuracy, precision, recall, f1, mcc, roc_auc,fpr])
      

        #print("X train after feature selection shape:", X_train_selected.shape)
        #scale needed for logistic regression, so we will scale the features using standard scaling. The scaled features will be used to train a logistic regression model, and the model will be evaluated using various metrics.  
        scaler = sc.create_scaler(X_train_selected,dataset_name)

        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)  

        # calling generic analyzing function
        accuracy, precision, recall, f1, mcc, roc_auc,fpr = an.analize_model_logostic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        feature_selection_results.append(["Logistic Regression",x, threshold if x != "select_k_best" else k, X_train_selected.shape[1], accuracy, precision, recall, f1, mcc, roc_auc,fpr])
      
    for result in feature_selection_results:
        print(result)
    """
    #------------------------------------------------------------------
    #Already find the best fs, dont run the top part takes forever....
    #------------------------------------------------------------------
    """
    print(f"regular pipe encoded before fs{X_train_encoded.shape}")
    #lest assume that the fs is good for 0.01 with random forest
    X_train_selected, X_test_selected, selected_columns, removed_columns, method = fs.apply_feature_selection(
            X_train_encoded,
            X_test_encoded,
            y_train,
            method="mutual_info",
            threshold= 0.015,
            k=20 #any number, mutul info need not k)
     )
    print(f"regular pipe encoded after fs{X_train_selected.shape}")
    """
    #---------------------------------------------------------
    #now hyper pramaetr on random forest, trying to find best 
    #parameters for that specific model
    #---------------------------------------------------------
    """
    best_model,best_parameter,best_cv_score = tune_random_forest(X_train_selected,y_train) 
    print("best model{0}, best param{1} best score {2}",best_model,best_parameter,best_cv_score)
  
    #----------------------------------------------------------------------------------------------------
    #now we try on test with the best model, and evaluate the performance of
    #the model using various metrics. The predicted labels will be compared with the true labels
    #to calculate accuracy, precision, recall, F1 score, Matthews correlation coefficient,
    #and ROC-AUC score. The confusion matrix and classification report will also be printed to provide a
    #detailed analysis of the model's performance.
    #-----------------------------------------------------------------------------------------------------
    
    #thevalues below are the ones I got from the code above, to save some time I use the values 

    best_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features=None,
    class_weight=None,
    n_jobs=1,
    random_state=42
    )

    best_model.fit(
    X_train_selected,
    y_train
    )
    
    y_pred = best_model.predict(X_test_selected)

    y_pred_proba = best_model.predict_proba(
        X_test_selected
    )[:, 1]

    metrics = an.calculate_metrics(
        y_test,
        y_pred,
        y_pred_proba
    )

    print("results with regular pipeline:",metrics)
    """
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

    live_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features=None,
    class_weight=None,
    n_jobs=1,
    random_state=42
)

    live_model.fit(
        X_train_live_selected,
        y_train_live
    )

    y_pred_live = live_model.predict(
        X_test_live_selected
    )

    y_pred_proba_live = live_model.predict_proba(
        X_test_live_selected
    )[:, 1]

    live_metrics = an.calculate_metrics(
        y_test_live,
        y_pred_live,
        y_pred_proba_live
    )

    print("Live-compatible model metrics (80/20) train only:")
    print(live_metrics)

    

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



if __name__ == "__main__":
    main()