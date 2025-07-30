import os

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()
USERNAME = 'admin'
PASSWORD = os.getenv("PASSWORD")

url = "http://127.0.0.1:8000/predict"
#url = "http://localhost:8000/predict"

auth = HTTPBasicAuth(USERNAME, PASSWORD)
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

response = requests.post(url, json=input_data, auth=auth)

if response.status_code == 200:
    print("Prediction: ", response.json())
else:
    print("Error:", response.status_code, response.json())