import pandas as pd
import pickle
from functools import lru_cache
from src.config import settings

def get_model():
    path = f"mlruns/{settings.EXPERIMENT_ID}/{settings.RUN_ID}/artifacts/model.pkl"
    with open("model/model.pkl", "rb") as f:
        return pickle.load(f)

def get_zipcode_features():
    return pd.read_csv("data/zipcode_demographics.csv").set_index("zipcode")
