import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from src.logger import log_prediction


load_dotenv()
USERNAME = 'admin'
PASSWORD = os.getenv("PASSWORD")

def get_prediction(url, input_data):
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    response = requests.post(url, json=input_data, auth=auth)
    if response.status_code == 200:
        log_prediction(input_data=response.json(), full=False)
        print("Prediction: ", response.json())
    else:
        print("Error:", response.status_code, response.json())

def get_predictions_full_input(url, data):
    auth = HTTPBasicAuth(USERNAME, PASSWORD)
    for i in range(len(data)):
        input_data = data.iloc[i].to_dict()
        response = requests.post(url, json=input_data, auth=auth)
        if response.status_code == 200:
            log_prediction(input_data=response.json(), full=True)
            print("Prediction: ", response.json())
        else:
            print("Error:", response.status_code, response.json())

if __name__ == "__main__":
    url = "http://127.0.0.1:8000/predict"
    url_full = "http://127.0.0.1:8000/predict_full"
    # url = "http://localhost:8000/predict"
    # url_full = "http://localhost:8000/predict_full"

    # Single prediction
    input_data = {
      "zipcode": 98042,
      "bedrooms": 4.0,
      "bathrooms": 1.0,
      "sqft_living": 1680.0,
      "sqft_lot": 5043.0,
      "floors": 1.5,
      "sqft_above": 1680.0,
      "sqft_basement": 1911.0
    }
    #get_prediction(url,input_data)

    # # Multiple predictions
    unseen_data = pd.read_csv('data/future_unseen_examples.csv')
    get_predictions_full_input(url_full, unseen_data)
