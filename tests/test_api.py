import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from fastapi.security import HTTPBasicCredentials
from api import app

# === Mock setup ===
mock_model = MagicMock()
mock_model.predict.return_value = np.array([500000.0])

mock_zipcode_data = pd.DataFrame({
    'ppltn_qty': [25495.0, 30905.0],
    'urbn_ppltn_qty': [25245.0, 30894.0],
    'sbrbn_ppltn_qty': [0.0, 0.0],
    'farm_ppltn_qty': [0.0, 0.0],
    'non_farm_qty': [250.0, 11.0],
    'medn_hshld_incm_amt': [60534.0, 36991.0],
    'medn_incm_per_prsn_amt': [24011.0, 18741.0],
    'hous_val_amt': [168400.0, 141500.0],
    'edctn_less_than_9_qty': [475.0, 1114.0],
    'edctn_9_12_qty': [1828.0, 3502.0],
    'edctn_high_schl_qty': [5293.0, 7859.0],
    'edctn_some_clg_qty': [5336.0, 6080.0],
    'edctn_assoc_dgre_qty': [1687.0, 1629.0],
    'edctn_bchlr_dgre_qty': [2725.0, 1903.0],
    'edctn_prfsnl_qty': [856.0, 785.0],
    'per_urbn': [99.0, 99.0],
    'per_sbrbn': [0.0, 0.0],
    'per_farm': [0.0, 0.0],
    'per_non_farm': [0.0, 0.0],
    'per_less_than_9': [1.0, 3.0],
    'per_9_to_12': [7.0, 11.0],
    'per_hsd': [20.0, 25.0],
    'per_some_clg': [20.0, 19.0],
    'per_assoc': [6.0, 5.0],
    'per_bchlr': [10.0, 6.0],
    'per_prfsnl': [3.0, 2.0],
    'zipcode': [98042, 98002]
}).set_index('zipcode')

client = TestClient(app)

# === Override authenticate ===
def override_authenticate():
    return HTTPBasicCredentials(username="user", password="pass")

@pytest.fixture(autouse=True)
def override_dependencies():
    from src.auth import authenticate
    app.dependency_overrides[authenticate] = override_authenticate
    yield
    app.dependency_overrides = {}

# === Fixtures ===
@pytest.fixture
def valid_payload():
    return {
      "zipcode": 98042,
      "bedrooms": 4.0,
      "bathrooms": 1.0,
      "sqft_living": 1680.0,
      "sqft_lot": 5043.0,
      "floors": 1.5,
      "sqft_above": 1680.0,
      "sqft_basement": 1911.0
    }

@patch("api.get_model", return_value=mock_model)
@patch("api.get_zipcode_features", return_value=mock_zipcode_data)
@patch("api.log_prediction")
def test_predict_endpoint(mock_log, mock_zip, mock_model_fn, valid_payload):
    response = client.post("/predict", json=valid_payload, auth=("user", "pass"))
    print("RESPONSE JSON:", response.json())
    #assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == [500000.0]
    assert "id" in data
    assert "timestamp" in data
    assert "model" in data

@patch("api.get_model", return_value=mock_model)
@patch("api.get_zipcode_features", return_value=mock_zipcode_data)
def test_predict_invalid_zipcode(mock_zip, mock_model_fn, valid_payload):
    valid_payload["zipcode"] = 99999
    response = client.post("/predict", json=valid_payload, auth=("user", "pass"))
    assert response.status_code == 404
    assert "Zipcode 99999 not found" in response.json()["detail"]
