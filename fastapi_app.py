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

#loading saved trained models
encoder_path = BASE_DIR / "Passive_IDS" / "models" / "live_encoder_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"
model_path = BASE_DIR / "Passive_IDS" / "models" / "random_forest_model_for_unswnb15_wo_swin_dwin_sttl_dttl.joblib"
columns_path = BASE_DIR / "Passive_IDS" / "models" / "live_selected_columns_for_unsenb15_wo_swin_dwin_sttl_dttl.joblib"

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
selected_columns = joblib.load(columns_path)

#working old version -keep for now....
"""
@app.post("/predict/unsw")
def predict_unsw(flow: dict):

    # one flow = one dataframe row
    df = pd.DataFrame([flow])

    encoded = encoder.transform(df)

    selected = encoded[selected_columns]

    prediction = model.predict(selected)[0]

    probability = model.predict_proba(selected)[0][1]

    return {
        "prediction": int(prediction),
        "label": "ATTACK" if prediction == 1 else "NORMAL",
        "attack_probability": float(probability)
    }
"""

@app.post("/predict/unsw_")
def predict_unsw_(flow: dict):

    try:
        print("\n--- NEW FLOW ---")
        print(flow)

        # One flow = one dataframe row
        df = pd.DataFrame([flow])

        print("DataFrame created")

        encoded = encoder.transform(df)

        print("Encoding completed")

        selected = encoded[selected_columns]

        print("Feature selection completed")

        prediction = model.predict(selected)[0]

        print("Prediction completed:", prediction)

        probability = model.predict_proba(selected)[0][1]

        print("Probability:", probability)

        # Safely get numeric values
        spkts = float(flow.get("spkts") or 0)
        dpkts = float(flow.get("dpkts") or 0)

        sbytes = float(flow.get("sbytes") or 0)
        dbytes = float(flow.get("dbytes") or 0)

        result = {
            "time": datetime.now().strftime("%H:%M:%S"),

            "protocol": str(flow.get("proto", "")),
            "service": str(flow.get("service", "")),
            "state": str(flow.get("state", "")),

            "packets": spkts + dpkts,
            "bytes": sbytes + dbytes,

            "prediction":
                "ATTACK" if prediction == 1 else "NORMAL",

            "probability": float(probability)
        }

        recent_predictions.appendleft(result)

        print("Added to dashboard")
        print("----------------")

        return {
            "prediction": int(prediction),
            "label":
                "ATTACK" if prediction == 1 else "NORMAL",
            "attack_probability": float(probability)
        }

    except Exception as e:

        print("\n*** PREDICTION ERROR ***")
        print(str(e))

        traceback.print_exc()

        print("************************\n")

        raise

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