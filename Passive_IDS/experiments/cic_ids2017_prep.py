import pandas as pd
import numpy as np
import gc
import joblib
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import preprocessing.loader as ld
from preprocessing.dataset_config_info import DATASET_CONFIG
import preprocessing.inspector as insp
import preprocessing.analyzer as an
import preprocessing.cleaner as cl
import preprocessing.splitter as splt
import preprocessing.preperrer as prep
import preprocessing.encoderer as enc
import preprocessing.featureselector as fs
import preprocessing.scaler as sc
from models.hyperparameter_tuner import tune_random_forest
"""
this is a pipeline pre for all the files for cic_ids2017
same logic as unsw_nb15
 cd Passive_IDS
 python -m experiments.cic_ids2017_prep

"""

#------------------------------------------------------------------
#first add al the files to a df (but add another column for source) 
#I dont wan to lose the data or make any assumptions
#------------------------------------------------------------------


def main() -> None:

    """load and analyze the UNSW-NB15
     dataset and the CIC-IDS2017 dataset,
      and print the reports for both datasets."""   
    
    dataset_name = "cic2017"

    config = DATASET_CONFIG[dataset_name]
    #get all files
    files=config["file_paths"]
    label_column = config["label_column"]
    #create df of dfs
    dfs=[]

    #now, trying to use the regular pipeline components 
    
    #-------------------------------------------------------------------------------
    #This part should only run once, after that we have only one cleaned merged file
    #-------------------------------------------------------------------------------
    """
    for file in files:

        df , report = an.analyze_dataset(
        file_path=file,
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
        )
        print("\nBefore cleaning:\n")
        an.print_report(report)

        #clean the df
        df_clean = cl.clean_dataset(df)
        
        #build new report
        report = insp.inspect_dataset(
            df=df_clean,
            label_column="Label" #cleaner strip spaces
        )

        print("\nAfter cleaning:\n")
        an.print_report(report)

        #add ref source file
        df_clean["source_file"] = file
        dfs.append(df_clean)

    #append to merged df
    df_merged = pd.concat(
     dfs,
     ignore_index=True
    )

    report = insp.inspect_dataset(
           df=df_merged,
           label_column="Label" #cleaner strip spaces
        )
        
    an.print_report(report)

    #at this point we have a cleaned merged version of abour 2Mil lines
    # saving for later, so no need to reclean every training
   
    df_merged.to_csv(
        "datasets/Processed/CIC-2017-MERGED/cic_2017.csv",
        index=False
    )
    """

    #------------------------------------------------------------------------------
    #from this point on, we only use the one merged file
    #-------------------------------------------------------------------------------
    
    df , report = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    an.print_report(report)

    #clean the df - no need we cleaned each file seperetly bbefore merge
    #df_clean = cl.clean_dataset(df)

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

    #no change expected, we dont have any categorical fields
    #in the training csv

    print("\nshape of X_train_encoded",X_train_encoded.shape)
   
    """
    # ----------------------------------------------------------
    # Free large objects that are no longer needed
    # ----------------------------------------------------------

    del df
    del X
    del X_train
    del X_test

    gc.collect()
    
    #only run if need the fs again

    #--------------------------------------------------------------------
    #Feature selection code, gets the best fs for the dataset
    #the machine cant handle 2 milion lines so we use sample of the 1.6
    #in the training section
    #--------------------------------------------------------------------

    FS_SAMPLE_SIZE = 250_000

    if len(y_train) > FS_SAMPLE_SIZE:

        splitter = sk.model_selection.StratifiedShuffleSplit(
            n_splits=1,
            train_size=FS_SAMPLE_SIZE,
            random_state=42
        )

        # We only need the selected indexes.
        # This avoids making a copy of the entire unused portion.
        fs_indexes, _ = next(
            splitter.split(
                np.zeros(len(y_train)),
                y_train
            )
        )

        # Support pandas DataFrame/Series or numpy arrays
        if hasattr(X_train_encoded, "iloc"):
            X_fs_pool = X_train_encoded.iloc[fs_indexes]
        else:
            X_fs_pool = X_train_encoded[fs_indexes]

        if hasattr(y_train, "iloc"):
            y_fs_pool = y_train.iloc[fs_indexes]
        else:
            y_fs_pool = np.asarray(y_train)[fs_indexes]

    else:
        X_fs_pool = X_train_encoded
        y_fs_pool = y_train


    print("\nFeature-selection pool X:", X_fs_pool.shape)
    print("\nFeature-selection pool Y:", y_fs_pool.shape)

    X_fs_train, X_fs_validation, y_fs_train, y_fs_validation = \
    sk.model_selection.train_test_split(
        X_fs_pool,
        y_fs_pool,
        test_size=0.20,
        random_state=42,
        stratify=y_fs_pool
    )

    print("FS training:", X_fs_train.shape)
    print("FS validation:", X_fs_validation.shape)

    # ==========================================================
    # FEATURE SELECTION EXPERIMENTS
    # ==========================================================

    feature_selection_results = []

    feature_selection_results.append([
        "testing-model",
        "method",
        "parameters",
        "features",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "fpr"
    ])


    feature_selection_methods = [
        {"method": "none"}
    ]


    # Variance
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


    # Correlation
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


    # Mutual information
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


    # Select K Best
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


    # ==========================================================
    # Find best feature selection
    # ==========================================================

    for method_config in feature_selection_methods:

        method_name = method_config["method"]

        threshold = method_config.get(
            "threshold",
            0.0
        )

        requested_k = method_config.get(
            "k",
            20
        )


        # ------------------------------------------------------
        # CIC may only have ~78 features.
        #
        # Prevent:
        # k=80 or k=100
        #
        # from failing if there are fewer features.
        # ------------------------------------------------------

        if method_name == "select_k_best":

            k = min(
                requested_k,
                X_fs_train.shape[1]
            )

        else:
            k = requested_k


        print(
            f"\nApplying feature selection:"
            f" method={method_name},"
            f" threshold={threshold},"
            f" k={k}"
        )


        # ------------------------------------------------------
        # IMPORTANT:
        #
        # Feature selection is now performed ONLY on the
        # smaller training sample.
        # ------------------------------------------------------

        (
            X_fs_train_selected,
            X_fs_validation_selected,
            selected_columns,
            removed_columns,
            method
        ) = fs.apply_feature_selection(

            X_fs_train,
            X_fs_validation,
            y_fs_train,

            method=method_name,
            threshold=threshold,
            k=k
        )

        #bypassing for savin due to ram issue
        if method_name == "correlation" and threshold == 0.90:
            joblib.dump(
                selected_columns,
                "models/cic2017_correlation_09_columns.joblib"
            )
            print("\nSaved correlation 0.90 selected columns:")
            print(len(selected_columns))
            print(selected_columns)



        print(
            "Selected feature count:",
            X_fs_train_selected.shape[1]
        )


        parameter_value = (
            k
            if method_name == "select_k_best"
            else threshold
        )


        # ======================================================
        # XGBOOST
        # ======================================================

        accuracy, precision, recall, f1, mcc, roc_auc, fpr = \
            an.analize_model_xgboost(
                X_fs_train_selected,
                X_fs_validation_selected,
                y_fs_train,
                y_fs_validation
            )


        feature_selection_results.append([
            "XGBoost",
            method_name,
            parameter_value,
            X_fs_train_selected.shape[1],
            accuracy,
            precision,
            recall,
            f1,
            mcc,
            roc_auc,
            fpr
        ])


        # ======================================================
        # RANDOM FOREST
        # ======================================================

        accuracy, precision, recall, f1, mcc, roc_auc, fpr = \
            an.analize_model_random_forest(
                X_fs_train_selected,
                X_fs_validation_selected,
                y_fs_train,
                y_fs_validation
            )


        feature_selection_results.append([
            "RandomForest",
            method_name,
            parameter_value,
            X_fs_train_selected.shape[1],
            accuracy,
            precision,
            recall,
            f1,
            mcc,
            roc_auc,
            fpr
        ])


        # ======================================================
        # LOGISTIC REGRESSION
        #
        # Scaling needed here.
        # ======================================================

        scaler = sc.create_scaler(
            X_fs_train_selected,
            dataset_name
        )

        X_fs_train_scaled = scaler.fit_transform(
            X_fs_train_selected
        )

        X_fs_validation_scaled = scaler.transform(
            X_fs_validation_selected
        )


        accuracy, precision, recall, f1, mcc, roc_auc, fpr = \
            an.analize_model_logostic_regression(
                X_fs_train_scaled,
                X_fs_validation_scaled,
                y_fs_train,
                y_fs_validation
            )


        feature_selection_results.append([
            "Logistic Regression",
            method_name,
            parameter_value,
            X_fs_train_selected.shape[1],
            accuracy,
            precision,
            recall,
            f1,
            mcc,
            roc_auc,
            fpr
        ])


        # ------------------------------------------------------
        # VERY IMPORTANT with CIC-IDS2017:
        # release temporary matrices after every experiment
        # ------------------------------------------------------

        del X_fs_train_selected
        del X_fs_validation_selected
        del X_fs_train_scaled
        del X_fs_validation_scaled
        del scaler

        gc.collect()


    # ==========================================================
    # Print feature-selection experiment results
    # ==========================================================

    print("\n\n==============================")
    print("FEATURE SELECTION RESULTS")
    print("==============================\n")

    for result in feature_selection_results:
        print(result)
    
    """
    #------------------------------------------------------------------------------------------
    #After runing the fs section i chose the
    # method="mutual_info" threshold=0.005 66
    # second best is 'correlation', 0.9, good results only 44 features- perfect for live sensor
    # get the fields from the selected fs and the sample
    #-------------------------------------------------------------------------------------------
    
    # Free some memory before selection

    print("Memory released before FS")

    del df
    del X
    del y
    del X_train
    del X_test

    gc.collect()

   
    #using the selected fs
    X_train_selected, X_test_selected, selected_columns, removed_columns, method = \
    fs.apply_feature_selection(
        X_train_encoded,
        X_test_encoded,
        y_train,
        method="correlation",
        threshold=0.9,
        k=20
    )

    print(f"Live pipe encoded selected after FS: {X_train_selected.shape}")
    print(f"Selected columns: {selected_columns}")
    #now save the selected features....

    print("Saving selected columns as joblib")

    joblib.dump(
    selected_columns,
    "models/cic2017/cic2017_correlation_090_selected_46_columns.joblib"
    )
    

    """
    #---------------------------------------------------------
    #now hyper pramaetr on random forest, trying to find best 
    #parameters for that specific model
    #I will do hyperparameter on a sample, it is too slow
    #---------------------------------------------------------
    _, X_tune, _, y_tune = train_test_split(
        X_train_selected,
        y_train,
        test_size=500000,
        stratify=y_train,
        random_state=42
    )
    print("Full training:", X_train_selected.shape)
    print("Tuning:", X_tune.shape)
    #best_model,best_parameter,best_cv_score = tune_random_forest(X_train_selected,y_train) 
    best_model, best_parameter, best_cv_score = tune_random_forest(
        X_tune,
        y_tune
    )
    print("best model{0}, best param{1} best score {2}",best_model,best_parameter,best_cv_score)

    """

    #------------------------------------------------------------------------------------------------
    #so we tried to optimize random forest clsifier with 200k and 500k, they gave us similar results
    # but not the same, so we will use both of them on the full 2mil lines and compare results
    #for this version i am using RandomForestClassifier, might add another one later
    #-------------------------------------------------------------------------------------------------
    """
    rf_200k = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="log2",
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )
          

    # Train on FULL training dataset using the 46 selected features
    rf_200k.fit(
        X_train_selected,
        y_train
    )

    # Test on the corresponding selected test features
    y_pred_200k = rf_200k.predict(
        X_test_selected
    )

    y_pred_proba_200k = rf_200k.predict_proba(
        X_test_selected
    )[:, 1]

    metrics_200k = an.calculate_metrics(
        y_test,
        y_pred_200k,
        y_pred_proba_200k
    )

    print("\nRF parameters selected using 200k tuning:")
    print(metrics_200k)
    """
    rf_500k = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        max_depth=40,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    )

    rf_500k.fit(
        X_train_selected,
        y_train
    )

    y_pred_500k = rf_500k.predict(
        X_test_selected
    )

    y_pred_proba_500k = rf_500k.predict_proba(
        X_test_selected
    )[:, 1]

    metrics_500k = an.calculate_metrics(
        y_test,
        y_pred_500k,
        y_pred_proba_500k
    )

    print("\nRF parameters selected using 500k tuning:")
    print(metrics_500k)

    #now save the model.....
    joblib.dump(
    rf_500k,
    "models/cic2017/cic2017_random_forest_46_columns.joblib"
    )

       
         
  

    


    
   
  
    




if __name__ == "__main__":
    main()