from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
#added for the dashboard
from collections import deque
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

recent_predictions = deque(maxlen=100)

#ini for dashboard load
capture_config={
    "running":False,
    "dataset":"cic2017",
    "model":"random_forest",
    "protocol":"all"
}

model_metrics = {
    "accuracy": 0.9499558014200576,
    "recall": 0.9728518161632242,
    "f1": 0.9635869452455547,
    "fpr": 0.09883928571428571
}

app = FastAPI()

#----------------------------------------------------------------------------
#added for dashboard (entire part for web gui)
#----------------------------------------------------------------------------
#add more access if needed (if on the network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:5001",
        "http://127.0.0.1:5001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##capture_running = True

##selected_model = "random_forest"

##recent_predictions = deque(
##    maxlen=100
##)

"""
class ModelRequest(BaseModel):
    model: str
"""

# accepting cic2017,random_forest,tcp 
class CaptureRequest(BaseModel):
    dataset: str
    model: str
    protocol: str


@app.get("/status")
def get_status():

    return {
        "capture_running":
            capture_config["running"],

        "dataset":
            capture_config["dataset"],    

        "model":
            capture_config["model"], 

        "protocol":
            capture_config["protocol"], 

        "metrics":
            model_metrics
    }

@app.post("/capture/start")
def start_capture(request:CaptureRequest):

    ##global capture_running

    capture_config["running"] = True

    capture_config["dataset"] = (request.dataset)

    capture_config["model"] = (request.model)

    capture_config["protocol"] = (request.protocol)

    print("Capture configuration:",capture_config)

    return {
        "status": "started",
        "config":capture_config
    }


@app.post("/capture/stop")
def stop_capture():

    ##global capture_running

    capture_config["running"] = False

    print("Capture stopped")

    return {
        "status": "stopped",
        "config":capture_config
    }

#entry point for the sensor to recive requested mode
@app.get("/capture/config")
def get_capture_config():
    return capture_config

"""
@app.post("/model")
def change_model(
    request: ModelRequest
):

    global selected_model

    selected_model = request.model

    print(
        "Selected model:",
        selected_model
    )

    return {
        "model":
            selected_model
    }

"""

@app.get("/predictions")
def get_predictions():

    print(
        "Dashboard requested predictions:",
        len(recent_predictions)
    )

    return list(recent_predictions)

#----------------------------------------------------------------------------
#End of gui part
#----------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------
#loading saved trained models for unsw
#--------------------------------------------------------------------------------------------------------------------

#staring with rf
unsw_rf_encoder_path = (
    BASE_DIR / "Passive_IDS" / "models" /
    "live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
)

unsw_rf_model_path = (
    BASE_DIR / "Passive_IDS" / "models" /
    "random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib"
)

unsw_rf_columns_path = (
    BASE_DIR / "Passive_IDS" / "models" /
    "live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
)

unsw_rf_encoder = joblib.load(
    unsw_rf_encoder_path
)

unsw_rf_model = joblib.load(
    unsw_rf_model_path
)

unsw_rf_selected_columns = joblib.load(
    unsw_rf_columns_path
)

#now xgboost
unsw_xgb_model_path = (
    BASE_DIR / "Passive_IDS" / "models" / "unswnb15" /
    "xgboost_live_unsw.joblib"
)

unsw_xgb_encoder_path = (
    BASE_DIR / "Passive_IDS" / "models" / "unswnb15" /
    "xgboost_live_encoder_unsw.joblib"
)

unsw_xgb_columns_path = (
    BASE_DIR / "Passive_IDS" / "models" / "unswnb15" /
    "xgboost_live_selected_columns_unsw.joblib"
)

unsw_xgb_model = joblib.load(
    unsw_xgb_model_path
)

unsw_xgb_encoder = joblib.load(
    unsw_xgb_encoder_path
)

unsw_xgb_selected_columns = joblib.load(
    unsw_xgb_columns_path
)

#just notification for future log
print(
    "UNSW Random Forest loaded with",
    len(unsw_rf_selected_columns),
    "features"
)

print(
    "UNSW XGBoost loaded with",
    len(unsw_xgb_selected_columns),
    "features"
)

#----------------------------
# isolation forest load
#----------------------------

# --------------------------------------------------------
# Load Isolation Forest model + encoder
# --------------------------------------------------------

isolation_model_path = (
    BASE_DIR /
    "Passive_IDS" /
    "models" /
    "unswnb15_isolation_forest" /
    "live_isolation_forest_v1.joblib"
)

isolation_encoder_path = (
    BASE_DIR /
    "Passive_IDS" /
    "models" /
    "unswnb15_isolation_forest" /
    "live_isolation_encoder_v1.joblib"
)

isolation_model = joblib.load(
    isolation_model_path
)

isolation_encoder = joblib.load(
    isolation_encoder_path
)

print(
    "Isolation Forest model loaded"
)


# --------------------------------------------------------
# Add next model if needed, please use same structure
# --------------------------------------------------------


# --------------------------------------------------------
# Updated version with Random Forest,
# XGBoost and Isolation Forest
# --------------------------------------------------------

@app.post("/predict/unsw")
def predict_unsw(flow: dict):

    # ------------------------------------------------------
    # Get selected model from dashboard configuration
    # ------------------------------------------------------

    selected_model = capture_config.get(
        "model",
        "random_forest"
    ).lower()


    # ------------------------------------------------------
    # Select model + matching encoder + matching columns
    # ------------------------------------------------------

    if selected_model == "random_forest":

        active_model = unsw_rf_model
        active_encoder = unsw_rf_encoder
        active_columns = unsw_rf_selected_columns

        model_type = "supervised"


    elif selected_model == "xgboost":

        active_model = unsw_xgb_model
        active_encoder = unsw_xgb_encoder
        active_columns = unsw_xgb_selected_columns

        model_type = "supervised"


    elif selected_model == "isolation_forest":

        active_model = isolation_model
        active_encoder = isolation_encoder

        # Isolation Forest was trained using
        # all encoded fields
        active_columns = None

        model_type = "unsupervised"


    else:

        return {
            "error": "Unsupported UNSW-NB15 model",
            "model": selected_model
        }


    # ------------------------------------------------------
    # Convert live flow to DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame(
        [flow]
    )


    # ------------------------------------------------------
    # Encode using encoder belonging to selected model
    # ------------------------------------------------------

    encoded = active_encoder.transform(
        df
    )


    # ------------------------------------------------------
    # Select exact columns belonging to selected model
    #
    # RF / XGBoost:
    #     use saved selected columns
    #
    # Isolation Forest:
    #     use all encoded fields
    # ------------------------------------------------------

    if active_columns is not None:

        selected = encoded[
            active_columns
        ]

    else:

        selected = encoded


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = active_model.predict(
        selected
    )[0]


    # ------------------------------------------------------
    # Supervised model prediction
    #
    # Random Forest / XGBoost
    # ------------------------------------------------------

    if model_type == "supervised":

        probability = active_model.predict_proba(
            selected
        )[0][1]

        prediction_label = (
            "ATTACK"
            if prediction == 1
            else "NORMAL"
        )

        anomaly_score = None


    # ------------------------------------------------------
    # Unsupervised model prediction
    #
    # Isolation Forest
    #
    # Isolation Forest prediction:
    #
    #  1 = NORMAL
    # -1 = ANOMALY
    #
    # decision_function:
    #
    # positive = more normal
    # negative = more anomalous
    # ------------------------------------------------------

    else:

        probability = None

        anomaly_score = active_model.decision_function(
            selected
        )[0]

        prediction_label = (
            "ANOMALY"
            if prediction == -1
            else "NORMAL"
        )


    # ------------------------------------------------------
    # Dashboard result
    # ------------------------------------------------------

    result = {

        "time":
            datetime.now().strftime(
                "%H:%M:%S"
            ),

        "dataset":
            "UNSW-NB15",

        "model":
            selected_model,

        "protocol":
            str(
                flow.get(
                    "proto",
                    ""
                )
            ),

        "service":
            str(
                flow.get(
                    "service",
                    ""
                )
            ),

        "state":
            str(
                flow.get(
                    "state",
                    ""
                )
            ),

        "packets":
            float(
                flow.get(
                    "spkts"
                ) or 0
            )
            +
            float(
                flow.get(
                    "dpkts"
                ) or 0
            ),

        "bytes":
            float(
                flow.get(
                    "sbytes"
                ) or 0
            )
            +
            float(
                flow.get(
                    "dbytes"
                ) or 0
            ),

        "prediction":
            prediction_label,

        # Random Forest / XGBoost only
        "probability":
            float(
                probability
            )
            if probability is not None
            else None,

        # Isolation Forest only
        "anomaly_score":
            float(
                anomaly_score
            )
            if anomaly_score is not None
            else None
    }


    recent_predictions.appendleft(
        result
    )


    # ------------------------------------------------------
    # Debug output
    # ------------------------------------------------------

    print(
        "UNSW DASHBOARD:",
        result
    )

    print(
        "UNSW model:",
        selected_model
    )

    print(
        "UNSW model type:",
        model_type
    )

    print(
        "UNSW feature count:",
        selected.shape[1]
    )


    if model_type == "supervised":

        print(
            "Attack probability:",
            probability
        )

    else:

        print(
            "Isolation Forest prediction:",
            prediction_label
        )

        print(
            "Isolation Forest anomaly score:",
            anomaly_score
        )


    print(
        "Stored predictions:",
        len(
            recent_predictions
        )
    )


    # ------------------------------------------------------
    # Return prediction to Network Agent
    # ------------------------------------------------------

    return {

        "prediction":
            int(
                prediction
            ),

        "label":
            prediction_label,

        "attack_probability":
            float(
                probability
            )
            if probability is not None
            else None,

        "anomaly_score":
            float(
                anomaly_score
            )
            if anomaly_score is not None
            else None,

        "model":
            selected_model
    }


"""
encoder_path = BASE_DIR / "Passive_IDS" / "models" / "live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
model_path = BASE_DIR / "Passive_IDS" / "models" / "random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib"
columns_path = BASE_DIR / "Passive_IDS" / "models" / "live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
selected_columns = joblib.load(columns_path)
"""


"""
@app.post("/predict/unsw")
def predict_unsw(flow: dict):

    df = pd.DataFrame([flow])

    encoded = encoder.transform(df)

    selected = encoded[selected_columns]

    prediction = model.predict(selected)[0]

    probability = model.predict_proba(selected)[0][1]

    result = {
        "time": datetime.now().strftime("%H:%M:%S"),

        "protocol": str(flow.get("proto", "")),
        "service": str(flow.get("service", "")),
        "state": str(flow.get("state", "")),

        "packets":
            float(flow.get("spkts") or 0) +
            float(flow.get("dpkts") or 0),

        "bytes":
            float(flow.get("sbytes") or 0) +
            float(flow.get("dbytes") or 0),

        "prediction":
            "ATTACK" if prediction == 1 else "NORMAL",

        "probability": float(probability)
    }

    recent_predictions.appendleft(result)

    print("DASHBOARD:", result)
    print("Stored predictions:", len(recent_predictions))

    return {
        "prediction": int(prediction),
        "label":
            "ATTACK" if prediction == 1 else "NORMAL",
        "attack_probability": float(probability)
    }

"""

@app.get("/test")
def testing():
    return {"status": "UP!"}

#--------------------------------------------------------
#load cic2017 model and columns
#--------------------------------------------------------

cic2017_rf_model_path = (
    BASE_DIR / "Passive_IDS" / "models" / "cic2017" /
    "cic2017_random_forest_46_columns.joblib"
)

cic2017_xgb_model_path = (
    BASE_DIR / "Passive_IDS" / "models" / "cic2017" /
    "cic2017_xgboost_46_columns.joblib"
)

# just the columns for both models
cic2017_columns_path = (
    BASE_DIR / "Passive_IDS" / "models" / "cic2017" /
    "cic2017_correlation_090_selected_46_columns.joblib"
)


cic2017_rf_model = joblib.load(
    cic2017_rf_model_path
)

cic2017_xgb_model = joblib.load(
    cic2017_xgb_model_path
)

cic2017_selected_columns = joblib.load(
    cic2017_columns_path
)


print(
    "CIC2017 Random Forest model loaded with",
    len(cic2017_selected_columns),
    "features"
)

print(
    "CIC2017 XGBoost model loaded with",
    len(cic2017_selected_columns),
    "features"
)

"""
cic2017_model_path = (
    BASE_DIR / "Passive_IDS" / "models" / "cic2017" /
    "cic2017_random_forest_46_columns.joblib"
)

cic2017_columns_path = (
    BASE_DIR / "Passive_IDS" / "models" / "cic2017" /
    "cic2017_correlation_090_selected_46_columns.joblib"
)

cic2017_model = joblib.load(cic2017_model_path)

cic2017_selected_columns = joblib.load(
    cic2017_columns_path
)

print(
    "CIC2017 model loaded with",
    len(cic2017_selected_columns),
    "features"
)
"""

@app.post("/predict/cic2017_works")
def predict_cic2017(payload: dict):

    # ------------------------------------------------------
    # Get features and metadata from request
    # ------------------------------------------------------

    features = payload.get("features", {})
    metadata = payload.get("metadata", {})


    # ------------------------------------------------------
    # Make sure all 46 required features were received
    # ------------------------------------------------------

    missing = [
        column
        for column in cic2017_selected_columns
        if column not in features
    ]

    if missing:
        return {
            "error": "Missing CIC-IDS2017 features",
            "missing": missing
        }


    # ------------------------------------------------------
    # Convert ONLY model features to DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame([features])


    # ------------------------------------------------------
    # Force exact same columns/order used during training
    # ------------------------------------------------------

    selected = df[cic2017_selected_columns]


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = cic2017_model.predict(selected)[0]

    probability = cic2017_model.predict_proba(
        selected
    )[0][1]


    # ------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------

    result = {
        "time": datetime.now().strftime("%H:%M:%S"),

        "dataset": "CIC-IDS2017",

        "src_ip": str(metadata.get("src_ip", "")),
        "src_port": metadata.get("src_port", ""),

        "dst_ip": str(metadata.get("dst_ip", "")),
        "dst_port": metadata.get("dst_port", ""),

        "protocol": str(metadata.get("protocol", "")),
        #added manually, not included in the 46 features selected
        "service": str(metadata.get("service", "")),
        "state": str(metadata.get("state", "")),

        "packets": float(
            metadata.get("packets") or 0
        ),

        "bytes": float(
            metadata.get("bytes") or 0
        ),

        "prediction":
            "ATTACK" if prediction == 1 else "NORMAL",

        "probability": float(probability)
    }


    recent_predictions.appendleft(result)

    print("CIC2017 DASHBOARD:", result)
    print(
        "Stored predictions:",
        len(recent_predictions)
    )


    # ------------------------------------------------------
    # Return prediction to Network Agent
    # ------------------------------------------------------

    return {
        "prediction": int(prediction),

        "label":
            "ATTACK" if prediction == 1 else "NORMAL",

        "attack_probability":
            float(probability)
    }


@app.post("/predict/cic2017")
def predict_cic2017(payload: dict):

    # ------------------------------------------------------
    # Get features and metadata from request
    # ------------------------------------------------------

    features = payload.get("features", {})
    metadata = payload.get("metadata", {})


    # ------------------------------------------------------
    # Make sure all 46 required features were received
    # ------------------------------------------------------

    missing = [
        column
        for column in cic2017_selected_columns
        if column not in features
    ]

    if missing:
        return {
            "error": "Missing CIC-IDS2017 features",
            "missing": missing
        }


    # ------------------------------------------------------
    # Convert ONLY model features to DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame([features])


    # ------------------------------------------------------
    # Force exact same columns/order used during training
    # ------------------------------------------------------

    selected = df[cic2017_selected_columns]


    # ------------------------------------------------------
    # Select model from dashboard configuration
    # ------------------------------------------------------

    selected_model = capture_config.get(
        "model",
        "random_forest"
    ).lower()


    if selected_model == "random_forest":

        active_model = cic2017_rf_model

    elif selected_model == "xgboost":

        active_model = cic2017_xgb_model

    else:

        return {
            "error": "Unsupported CIC-IDS2017 model",
            "model": selected_model
        }


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    prediction = active_model.predict(
        selected
    )[0]

    probability = active_model.predict_proba(
        selected
    )[0][1]


    # ------------------------------------------------------
    # Dashboard data
    # ------------------------------------------------------

    result = {
        "time": datetime.now().strftime("%H:%M:%S"),

        "dataset": "CIC-IDS2017",

        # Model that made this prediction
        "model": selected_model,

        "src_ip": str(metadata.get("src_ip", "")),
        "src_port": metadata.get("src_port", ""),

        "dst_ip": str(metadata.get("dst_ip", "")),
        "dst_port": metadata.get("dst_port", ""),

        "protocol": str(metadata.get("protocol", "")),

        # Added manually, not included in the 46 features selected
        "service": str(metadata.get("service", "")),
        "state": str(metadata.get("state", "")),

        "packets": float(
            metadata.get("packets") or 0
        ),

        "bytes": float(
            metadata.get("bytes") or 0
        ),

        "prediction":
            "ATTACK" if prediction == 1 else "NORMAL",

        "probability": float(probability)
    }


    recent_predictions.appendleft(result)

    print("CIC2017 DASHBOARD:", result)

    print(
        "Model used:",
        selected_model
    )

    print(
        "Stored predictions:",
        len(recent_predictions)
    )


    # ------------------------------------------------------
    # Return prediction to Network Agent
    # ------------------------------------------------------

    return {
        "prediction": int(prediction),

        "label":
            "ATTACK" if prediction == 1 else "NORMAL",

        "attack_probability":
            float(probability),

        "model":
            selected_model
    }