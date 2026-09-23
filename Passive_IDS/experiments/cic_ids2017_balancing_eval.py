import pandas as pd
import numpy as np
import gc
import joblib
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os
from sklearn.utils import resample

# customized code section here
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

from models.hyperparameter_tuner import (
    tune_random_forest,
    tune_xgboost
)


"""
This experiment tests whether the class imbalance in the merged
CIC-IDS2017 dataset contributes to the very high XGBoost results.

The experiment:

1. Loads the merged CIC-IDS2017 dataset.
2. Performs the same 80/20 train/test split.
3. Uses the same correlation feature selection used by the live system.
4. Balances ONLY the training set to 50% BENIGN / 50% ATTACK.
5. Trains XGBoost using the same tuned parameters.
6. Evaluates using the ORIGINAL untouched test set.

To run:

cd Passive_IDS
python -m experiments.cic_ids2017_prep_XGBOOST
"""


def main() -> None:

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    dataset_name = "cic2017"

    config = DATASET_CONFIG[dataset_name]

    label_column = config["label_column"]


    # ----------------------------------------------------------
    # Load merged CIC-IDS2017 dataset
    # ----------------------------------------------------------

    df, report = an.analyze_dataset(
        file_path=config["file_path"],
        dataset_name=dataset_name,
        label_column=config["label_column"],
        columns=config["columns"]
    )

    an.print_report(report)


    # ----------------------------------------------------------
    # Prepare X and y
    #
    # Binary:
    #
    # 0 = BENIGN
    # 1 = ATTACK
    # ----------------------------------------------------------

    X, y = prep.prepare_xy(
        df,
        dataset_name
    )


    # ----------------------------------------------------------
    # Split dataset
    #
    # 80% training
    # 20% testing
    #
    # IMPORTANT:
    # The test set will remain untouched for this experiment.
    # ----------------------------------------------------------

    X_train, X_test, y_train, y_test = \
        splt.split_dataset(
            X,
            y,
            dataset_name,
            test_size=0.2,
            random_state=42
        )


    print(
        "\nShape of X_train:",
        X_train.shape
    )


    # ----------------------------------------------------------
    # Encoding
    #
    # CIC-IDS2017 currently has no categorical fields,
    # so no dimensional change is expected.
    # ----------------------------------------------------------

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
        "\nShape of X_train_encoded:",
        X_train_encoded.shape
    )


    # ----------------------------------------------------------
    # Free memory
    # ----------------------------------------------------------

    print("\nMemory released before FS")

    del df
    del X
    del y
    del X_train
    del X_test

    gc.collect()


    # ----------------------------------------------------------
    # Feature Selection
    #
    # Use the SAME feature-selection configuration used
    # for the current live CIC-IDS2017 model.
    #
    # correlation threshold = 0.9
    #
    # Expected result: 46 features
    # ----------------------------------------------------------

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
        method="correlation",
        threshold=0.9,
        k=20
    )


    print(
        "\nLive pipe encoded selected after FS:",
        X_train_selected.shape
    )

    print(
        "\nSelected columns:"
    )

    print(
        selected_columns
    )


    # ==========================================================
    # EXPERIMENT: BALANCED 50/50 TRAINING DATA
    # ==========================================================

    print(
        "\n=============================================="
    )

    print(
        "BALANCED TRAINING EXPERIMENT"
    )

    print(
        "=============================================="
    )


    # ----------------------------------------------------------
    # Reset indexes
    #
    # This makes sure X and y use the same sequential index.
    # ----------------------------------------------------------

    X_train_balance = \
        X_train_selected.reset_index(
            drop=True
        )

    y_train_balance = \
        y_train.reset_index(
            drop=True
        )


    # ----------------------------------------------------------
    # Show ORIGINAL training distribution
    # ----------------------------------------------------------

    print(
        "\nOriginal training distribution:"
    )

    print(
        y_train_balance.value_counts()
    )


    print(
        "\nOriginal training percentages:"
    )

    print(
        y_train_balance.value_counts(
            normalize=True
        ) * 100
    )


    # ----------------------------------------------------------
    # Attach target to features temporarily
    #
    # This prevents feature/label alignment problems while
    # sampling and shuffling.
    # ----------------------------------------------------------

    train_data = X_train_balance.copy()

    train_data["target"] = \
        y_train_balance


    # ----------------------------------------------------------
    # Separate BENIGN and ATTACK
    # ----------------------------------------------------------

    normal_data = train_data[
        train_data["target"] == 0
    ]

    attack_data = train_data[
        train_data["target"] == 1
    ]


    print(
        "\nNormal training rows:",
        len(normal_data)
    )

    print(
        "Attack training rows:",
        len(attack_data)
    )


    # ----------------------------------------------------------
    # Randomly undersample BENIGN
    #
    # Number of BENIGN rows will be made equal to the
    # number of ATTACK rows.
    #
    # Example:
    #
    # Before:
    # BENIGN  1,700,000
    # ATTACK    340,000
    #
    # After:
    # BENIGN    340,000
    # ATTACK    340,000
    # ----------------------------------------------------------

    normal_balanced = resample(
        normal_data,
        replace=False,
        n_samples=len(attack_data),
        random_state=42
    )


    # ----------------------------------------------------------
    # Combine balanced BENIGN + ATTACK
    # ----------------------------------------------------------

    balanced_data = pd.concat(
        [
            normal_balanced,
            attack_data
        ],
        ignore_index=True
    )


    # ----------------------------------------------------------
    # Shuffle balanced training dataset
    # ----------------------------------------------------------

    balanced_data = balanced_data.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )


    # ----------------------------------------------------------
    # Separate features and labels again
    # ----------------------------------------------------------

    y_train_balanced = balanced_data[
        "target"
    ]

    X_train_balanced = balanced_data.drop(
        columns=["target"]
    )


    # ----------------------------------------------------------
    # Show BALANCED training distribution
    # ----------------------------------------------------------

    print(
        "\nBalanced training distribution:"
    )

    print(
        y_train_balanced.value_counts()
    )


    print(
        "\nBalanced training percentages:"
    )

    print(
        y_train_balanced.value_counts(
            normalize=True
        ) * 100
    )


    print(
        "\nBalanced X_train shape:",
        X_train_balanced.shape
    )


    # ----------------------------------------------------------
    # Free some memory
    # ----------------------------------------------------------

    del train_data
    del normal_data
    del attack_data
    del normal_balanced
    del X_train_balance
    del y_train_balance

    gc.collect()


    # ==========================================================
    # TRAIN XGBOOST
    #
    # IMPORTANT:
    #
    # These are EXACTLY the same parameters as the tuned
    # XGBoost model trained on the original dataset.
    #
    # The ONLY major experimental difference is that the
    # training data is now balanced 50/50.
    # ==========================================================

    xgb_balanced = XGBClassifier(

        subsample=1.0,

        n_estimators=200,

        min_child_weight=5,

        max_depth=7,

        learning_rate=0.2,

        gamma=0.3,

        colsample_bytree=0.7,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=2
    )


    print(
        "\nTraining XGBoost on BALANCED training data..."
    )


    xgb_balanced.fit(
        X_train_balanced,
        y_train_balanced
    )


    # ==========================================================
    # TEST USING ORIGINAL UNTOUCHED TEST SET
    #
    # DO NOT BALANCE THE TEST SET.
    # ==========================================================

    print(
        "\nTesting balanced model on ORIGINAL test set..."
    )


    y_pred_balanced = \
        xgb_balanced.predict(
            X_test_selected
        )


    y_proba_balanced = \
        xgb_balanced.predict_proba(
            X_test_selected
        )[:, 1]


    # ==========================================================
    # CALCULATE METRICS
    # ==========================================================

    metrics_balanced = \
        an.calculate_metrics(
            y_test,
            y_pred_balanced,
            y_proba_balanced
        )


    # ==========================================================
    # RESULTS
    # ==========================================================

    print(
        "\n=============================================="
    )

    print(
        "BALANCED XGBOOST RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        "\nAccuracy, Precision, Recall, F1, "
        "MCC, ROC-AUC, FPR"
    )

    print(
        metrics_balanced
    )


    # ----------------------------------------------------------
    # Current original XGBoost result
    #
    # This is here only to make comparison easier.
    # ----------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "ORIGINAL XGBOOST RESULT FOR COMPARISON"
    )

    print(
        "=============================================="
    )


    print(
        "Accuracy : 0.9991467908452019"
    )

    print(
        "Precision: 0.9966813251245968"
    )

    print(
        "Recall   : 0.9981678958989054"
    )

    print(
        "F1       : 0.997424056612077"
    )

    print(
        "MCC      : 0.9969131767374532"
    )

    print(
        "ROC-AUC  : 0.9999751697500551"
    )

    print(
        "FPR      : 0.0006590898504820904"
    )


    print(
        "\n=============================================="
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()