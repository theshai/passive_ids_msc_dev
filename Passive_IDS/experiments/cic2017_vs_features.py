import joblib
import pandas as pd
import numpy as np

selected_columns = joblib.load(
    "models/cic2017/cic2017_correlation_090_selected_46_columns.joblib"
)

#features from live sensor
features={'Destination Port': 80,
    'Flow Duration': 19000000.0,
    'Total Fwd Packets': 5,
    'Total Length of Fwd Packets': 500,
    'Fwd Packet Length Max': 100,
    'Fwd Packet Length Min': 100,
    'Fwd Packet Length Mean': 100.0,
    'Bwd Packet Length Max': 0,
    'Bwd Packet Length Min': 0,
    'Flow Bytes/s': 26.31578947368421,
    'Flow Packets/s': 0.2631578947368421,
    'Flow IAT Mean': 4750000.0,
    'Flow IAT Std': np.float64(3344772.040064913),
    'Flow IAT Min': 1000000.0,
    'Fwd IAT Min': 1000000.0,
    'Bwd IAT Total': 0,
    'Bwd IAT Mean': 0,
    'Bwd IAT Std': 0,
    'Bwd IAT Max': 0,
    'Fwd PSH Flags': 0,
    'Bwd PSH Flags': 0,
    'Fwd URG Flags': 0,
    'Bwd URG Flags': 0,
    'Fwd Header Length': 100,
    'Bwd Header Length': 0,
    'Bwd Packets/s': 0.0,
    'min_seg_size_forward': 20,
    'Min Packet Length': 100,
    'FIN Flag Count': 0,
    'RST Flag Count': 0,
    'PSH Flag Count': 0,
    'ACK Flag Count': 5,
    'URG Flag Count': 0,
    'Down/Up Ratio': 0.0,
    'Init_Win_bytes_forward': 8192,
    'Init_Win_bytes_backward': 0,
    'Fwd Avg Bytes/Bulk': 0,
    'Fwd Avg Packets/Bulk': 0,
    'Fwd Avg Bulk Rate': 0,
    'Bwd Avg Bytes/Bulk': 0,
    'Bwd Avg Packets/Bulk': 0,
    'Bwd Avg Bulk Rate': 0,
    'Active Mean': 1500000.0,
    'Active Std': 500000.0,
    'Active Max': 2000000.0,
    'Idle Std': 1000000.0
    }


def prepare_cic2017_for_model(features):

    missing = [
        col for col in selected_columns
        if col not in features
    ]

    if missing:
        raise ValueError(
            f"Missing CIC-IDS2017 features: {missing}"
        )

    # Create one-row DataFrame
    X_live = pd.DataFrame([features])

    # Force exact same features AND order as training
    X_live = X_live[selected_columns]

    return X_live


X_live = prepare_cic2017_for_model(features)

print(X_live.shape)
print(X_live.columns.tolist())

model = joblib.load(
    "models/cic2017/cic2017_random_forest_46_columns.joblib"
)

prediction = model.predict(X_live)[0]
probabilities = model.predict_proba(X_live)[0]

print("\n----------------CIC prediction--------")
print("Prediction:", prediction)
print("Label:", "ATTACK" if prediction == 1 else "NORMAL")
print("Normal probability:", probabilities[0])
print("Attack probability:", probabilities[1])
print("-----------------------------------------")