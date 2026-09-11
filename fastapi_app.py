from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

#loading saved trained models
encoder_path = BASE_DIR / "Passive_IDS" / "models" / "live_encoder_for_unsenb15.joblib"
model_path = BASE_DIR / "Passive_IDS" / "models" / "random_forest_model_for_unswnb15.joblib"
columns_path = BASE_DIR / "Passive_IDS" / "models" / "live_selected_columns_for_unsenb15.joblib"

encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
selected_columns = joblib.load(columns_path)

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
    
@app.get("/test")
def testing():
    return {"status": "received"}
