from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasicCredentials
from src.auth import authenticate
from src.validation import InputFeatures, FullInputFeatures
from src.logger import log_prediction
from src.utils import get_model, get_zipcode_features
from datetime import datetime
from src.config import settings
import numpy as np
import uuid

app = FastAPI()

@app.post("/predict")
def predict(data: InputFeatures, credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        zipcode_row = get_zipcode_features().loc[data.zipcode]
        user_features = [
            data.bedrooms, data.bathrooms, data.sqft_living, data.sqft_lot,
            data.floors, data.sqft_above, data.sqft_basement
        ]
        input_features = np.array(user_features + zipcode_row.tolist()).reshape(1, -1)
        prediction = get_model().predict(input_features)

        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()


        return {
            "id": request_id,
            "timestamp": timestamp,
            "prediction": prediction.tolist(),
            "model": {"experiment_id":settings.EXPERIMENT_ID,"run_id":settings.RUN_ID},
            "features":data.dict()
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Zipcode {data.zipcode} not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_full")
def predict_full(data: FullInputFeatures, credentials: HTTPBasicCredentials = Depends(authenticate)):
    try:
        zipcode_row = get_zipcode_features().loc[data.zipcode]
        selected_features = [
            data.bedrooms, data.bathrooms, data.sqft_living, data.sqft_lot,
            data.floors, data.sqft_above, data.sqft_basement
        ]
        input_features = np.array(selected_features + zipcode_row.tolist()).reshape(1, -1)
        prediction = get_model().predict(input_features)

        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        return {
            "id": request_id,
            "timestamp": timestamp,
            "prediction": prediction.tolist(),
            "model": {"experiment_id": settings.EXPERIMENT_ID, "run_id": settings.RUN_ID},
            "features": data.dict()
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Zipcode {data.zipcode} not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))