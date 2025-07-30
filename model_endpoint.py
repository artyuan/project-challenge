import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from pydantic import BaseModel
import uuid
from datetime import datetime
import shap
import pandas as pd
import pickle
import os

load_dotenv()

USERNAME = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")

if not USERNAME or not PASSWORD:
    raise ValueError("Environment variables USERNAME and PASSWORD must be set.")



app = FastAPI()
security = HTTPBasic()

LOG_FILE = "data/prediction_logs.csv"

def get_model():
    with open("model/model.pkl","rb") as f:
        model = pickle.load(f)
    return model

def get_zipcode_features():
    return pd.read_csv("data/zipcode_demographics.csv").set_index("zipcode")

class InputFeatures(BaseModel):
    zipcode: int
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USERNAME or credentials.password != PASSWORD:
        print("Received username:", credentials.username)
        print("Expected username:", USERNAME)
        detail= {
            "Received username": credentials.username,
            "Expected username": USERNAME,
            'Unauthorized': 'Unauthorized'
        }
        raise HTTPException(status_code=401, detail=detail)
    return credentials
@app.post("/predict")
def predict(data: InputFeatures, credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        zipcode_features = get_zipcode_features()
        zipcode_row = zipcode_features.loc[data.zipcode]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Zipcode {data.zipcode} not found.")
    try:
        user_features = [
            data.bedrooms,
            data.bathrooms,
            data.sqft_living,
            data.sqft_lot,
            data.floors,
            data.sqft_above,
            data.sqft_basement
        ]
        model = get_model()
        # Combine with zipcode-level features
        input_features = np.array(user_features + zipcode_row.tolist()).reshape(1, -1)
        prediction = model.predict(input_features)

        # Generate metadata
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Log data
        log_data = {
            "id": request_id,
            "timestamp": timestamp,
            "zipcode": data.zipcode,
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "sqft_living": data.sqft_living,
            "sqft_lot": data.sqft_lot,
            "floors": data.floors,
            "sqft_above": data.sqft_above,
            "sqft_basement": data.sqft_basement,
            "prediction": prediction[0]
        }

        # Convert to DataFrame and append to CSV
        df_log = pd.DataFrame([log_data])

        if not os.path.isfile(LOG_FILE):
            df_log.to_csv(LOG_FILE, index=False)
        else:
            df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
        return {
            "id": request_id,
            "timestamp": timestamp,
            "prediction": prediction.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=(str(e)))
