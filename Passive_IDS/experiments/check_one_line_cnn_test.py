import joblib
import pandas as pd


# ============================================================
# 1. Load exactly the same objects used by FastAPI
# ============================================================

encoder = joblib.load("models/live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib")
model = joblib.load("models/random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib")

# If you saved selected_columns separately:
selected_columns = joblib.load(
    "models/live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
)


# ============================================================
# 2. CNN live flow
# ============================================================

ive_sample = {
    "dur": 26.983758,
    "proto": "tcp",
    "service": "-",
    "state": "FIN",
    "spkts": 68,
    "dpkts": 229,
    "sbytes": 4923,
    "dbytes": 340246,
    "rate": 11.00662109406703,
    #"dttl": 121,
    "smean": 72.3970588235294,
    "dmean": 1485.7903930131004,
    "sinpkt": 402.74234328358204,
    "dinpkt": 117.5579956140351,
    "sjit": 2102.779964636061,
    "djit": 1154.81604804538,
    "sloss": 0,
    "dloss": 8,
    "sload": 1459.5446638677977,
    "dload": 100874.31113190386,
    "stcpb": 916922441,
    "dtcpb": 2344104176,
    "synack": 0.160463,
    "ackdat": 0.001851,
    "tcprtt": 0.162314,
    "is_sm_ips_ports": 0,
    "response_body_len": 0
}


# ============================================================
# 3. Convert live record to DataFrame
# ============================================================

X_live = pd.DataFrame([ive_sample])


# ============================================================
# 4. Apply EXACT SAME encoder used during training
# ============================================================

X_live_encoded = encoder.transform(X_live)

# If encoder was configured using:
# encoder.set_output(transform="pandas")
# this should already be a DataFrame.
#
# Otherwise convert it.

if not isinstance(X_live_encoded, pd.DataFrame):

    encoded_columns = encoder.get_feature_names_out()

    X_live_encoded = pd.DataFrame(
        X_live_encoded,
        columns=encoded_columns
    )


print("\n================================================")
print("ENCODED live FEATURES")
print("================================================")

print(X_live_encoded.T.to_string())


# ============================================================
# 5. Apply feature-selection columns
# ============================================================

X_live_selected = X_live_encoded[selected_columns]


print("\n================================================")
print("SELECTED live FEATURES")
print("================================================")

print(X_live_selected.T.to_string())


# ============================================================
# 6. Prediction
# ============================================================

prediction = model.predict(X_live_selected)[0]

probabilities = model.predict_proba(
    X_live_selected
)[0]

attack_probability = probabilities[1]


print("\n================================================")
print("PREDICTION")
print("================================================")

print("Prediction:", prediction)
print("Normal probability:", probabilities[0])
print("Attack probability:", probabilities[1])


# ============================================================
# 7. Random Forest GLOBAL feature importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": selected_columns,
    "line_value": X_live_selected.iloc[0].values,
    "importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\n================================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("================================================")

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 8. Top 15 only
# ============================================================

print("\n================================================")
print("TOP 15 MOST IMPORTANT FEATURES")
print("================================================")

print(
    importance_df.head(15).to_string(
        index=False
    )
)