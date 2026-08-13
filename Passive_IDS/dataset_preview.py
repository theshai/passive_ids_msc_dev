# simple test loading csv
import pandas as pd
import numpy as np
import sklearn as sk
from pandas import read_csv
from pandas import set_option
import matplotlib.pyplot as plt
import sklearn as sk
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

#the  pahth to the csv file (run root@021063d16b76:/workspace/Passive_IDS# python dataset_preview.py )
#file_path_KDD = 'datasets/RAW/KDD-NSL/KDDTrain+.txt'
file_path_KDD = 'datasets/RAW/KDD-NSL/KDDTest+.txt'
file_path_CIC2017 = 'datasets/RAW/CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
file_path_UNSW = 'datasets/RAW/UNSW-NB15/UNSW_NB15_training-set.csv'

file_path_UNSW_processed = 'datasets/Processed/UNSW-NB15/UNSW_NB15_training_processed.csv'
df_unsw_processed =read_csv(file_path_UNSW_processed, header=0, sep=",")

"""
# set the labels for KDD dataset - it is unlabeled, so we need to provide the column names
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty"
]

#df_kdd=read_csv(file_path_KDD, names=columns,header=None, sep="\t")
#df_kdd=read_csv(file_path_KDD, names=columns,header=None, sep=",")
#print(df_kdd.info())

df_unsw=read_csv(file_path_UNSW, header=0, sep=",")
print(df_unsw.info())

# Copy the original dataframe
df_unsw_processed = df_unsw.copy()

# Categorical columns to One-Hot Encode
categorical_columns = [
    "proto",
    "service",
    "state"
]

for column in categorical_columns:

    # Find the original position of the column
    position = df_unsw_processed.columns.get_loc(column)

    # Create encoder
    encoder = sk.preprocessing.OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Encode the column
    encoded_data = encoder.fit_transform(
        df_unsw_processed[[column]]
    )

    # Get new column names
    encoded_columns = encoder.get_feature_names_out([column])

    # Create DataFrame
    df_encoded = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=df_unsw_processed.index
    )

    # Remove original column
    df_unsw_processed.drop(columns=[column], inplace=True)

    # Insert encoded columns in the same location
    for i, new_column in enumerate(encoded_columns):
        df_unsw_processed.insert(
            position + i,
            new_column,
            df_encoded[new_column]
        )

# --------------------------------------------------
# attack_cat
# --------------------------------------------------
# Leave as text if you want the attack names.
# If you don't need it for training, remove it:

df_unsw_processed.drop(columns=["attack_cat"], inplace=True)

# --------------------------------------------------
# label
# --------------------------------------------------
# Already numeric (0 = Normal, 1 = Attack)
df_unsw_processed["label"] = df_unsw_processed["label"].astype(int)

# Check column order
for i, column in enumerate(df_unsw_processed.columns):
    print(i, column)

# Save processed dataset
output_file = "datasets/Processed/UNSW-NB15/UNSW_NB15_training_processed.csv"

df_unsw_processed.to_csv(
    output_file,
    index=False
)

print("Saved:", output_file)
"""

print(df_unsw_processed.dtypes)

print("is null", df_unsw_processed.isnull().sum())

print("duplicated", df_unsw_processed.duplicated().sum())

X=df_unsw_processed.drop(columns=["label"])
y=df_unsw_processed["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)  


print(y.value_counts())
print("y value counts normalized:", y.value_counts(normalize=True))

# skiping class balance for now, we will handle it later in the training phase

# do split train test before scaling, to avoid data leakage
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)

scaler = sk.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  

# first model
lr_model = sk.linear_model.LogisticRegression(max_iter=1000,random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained.....")

y_pred = lr_model.predict(X_test_scaled)
# for roc-auc, we need the predicted probabilities for the positive class
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

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
print("Saving the model and scaler to disk...")
# Save the model and scaler to disk
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model and scaler saved.")    
"""