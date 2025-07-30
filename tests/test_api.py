import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys

# Mock data for testing
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
    'zipcode': [98001, 98002]
}).set_index('zipcode')

mock_model = MagicMock()
mock_model.predict.return_value = np.array([500000.0])

class TestModelEndpoint:
    @classmethod
    def setup_class(cls):
        """Set up test environment before importing the app"""
        # Set environment variables before importing the module
        os.environ['USER'] = 'testuser'
        os.environ['PASSWORD'] = 'testpass'

        # Import the app after setting environment variables
        from model_endpoint import app
        cls.client = TestClient(app)

    def test_predict_success(self):
        """Test successful prediction with valid credentials and data"""
        # Mock the functions at the module level
        with patch('model_endpoint.get_model', return_value=mock_model), \
                patch('model_endpoint.get_zipcode_features', return_value=mock_zipcode_data), \
                patch('os.path.isfile', return_value=False), \
                patch('pandas.DataFrame.to_csv') as mock_to_csv:
            # Test data
            test_data = {
                "zipcode": 98001,
                "bedrooms": 3.0,
                "bathrooms": 2.5,
                "sqft_living": 2000.0,
                "sqft_lot": 5000.0,
                "floors": 2.0,
                "sqft_above": 1800.0,
                "sqft_basement": 200.0
            }

            response = self.client.post(
                "/predict",
                json=test_data,
                auth=("testuser", "testpass")
            )

            # Debug: Print response details if it fails
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert "timestamp" in data
            assert "prediction" in data
            assert isinstance(data["prediction"], list)
            assert len(data["prediction"]) == 1

    def test_predict_unauthorized(self):
        """Test prediction with invalid credentials"""
        test_data = {
            "zipcode": 98001,
            "bedrooms": 3.0,
            "bathrooms": 2.5,
            "sqft_living": 2000.0,
            "sqft_lot": 5000.0,
            "floors": 2.0,
            "sqft_above": 1800.0,
            "sqft_basement": 200.0
        }

        response = self.client.post(
            "/predict",
            json=test_data,
            auth=("wronguser", "wrongpass")
        )

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    def test_predict_invalid_zipcode(self):
        """Test prediction with non-existent zipcode"""
        with patch('model_endpoint.get_zipcode_features', return_value=mock_zipcode_data):
            test_data = {
                "zipcode": 99999,  # Non-existent zipcode
                "bedrooms": 3.0,
                "bathrooms": 2.5,
                "sqft_living": 2000.0,
                "sqft_lot": 5000.0,
                "floors": 2.0,
                "sqft_above": 1800.0,
                "sqft_basement": 200.0
            }

            response = self.client.post(
                "/predict",
                json=test_data,
                auth=("testuser", "testpass")
            )

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert "Zipcode 99999 not found" in data["detail"]

    def test_predict_invalid_input_data(self):
        """Test prediction with invalid input data types"""
        test_data = {
            "zipcode": "invalid",
            "bedrooms": "three",
            "bathrooms": 2.5,
            "sqft_living": 2000.0,
            "sqft_lot": 5000.0,
            "floors": 2.0,
            "sqft_above": 1800.0,
            "sqft_basement": 200.0
        }

        response = self.client.post(
            "/predict",
            json=test_data,
            auth=("testuser", "testpass")
        )

        assert response.status_code == 422  # Validation error

    def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        test_data = {
            "zipcode": 98001,
            "bedrooms": 3.0,
        }

        response = self.client.post(
            "/predict",
            json=test_data,
            auth=("testuser", "testpass")
        )

        assert response.status_code == 422

    def test_predict_model_error(self):
        """Test prediction when model raises an exception"""
        error_model = MagicMock()
        error_model.predict.side_effect = Exception("Model error")

        with patch('model_endpoint.get_model', return_value=error_model), \
                patch('model_endpoint.get_zipcode_features', return_value=mock_zipcode_data):
            test_data = {
                "zipcode": 98001,
                "bedrooms": 3.0,
                "bathrooms": 2.5,
                "sqft_living": 2000.0,
                "sqft_lot": 5000.0,
                "floors": 2.0,
                "sqft_above": 1800.0,
                "sqft_basement": 200.0
            }

            response = self.client.post(
                "/predict",
                json=test_data,
                auth=("testuser", "testpass")
            )

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            assert "Model error" in data["detail"]

    def test_predict_logging(self):
        """Test that predictions are logged correctly"""
        with patch('model_endpoint.get_model', return_value=mock_model), \
                patch('model_endpoint.get_zipcode_features', return_value=mock_zipcode_data), \
                patch('os.path.isfile', return_value=False), \
                patch('pandas.DataFrame.to_csv') as mock_to_csv:
            test_data = {
                "zipcode": 98001,
                "bedrooms": 3.0,
                "bathrooms": 2.5,
                "sqft_living": 2000.0,
                "sqft_lot": 5000.0,
                "floors": 2.0,
                "sqft_above": 1800.0,
                "sqft_basement": 200.0
            }

            response = self.client.post(
                "/predict",
                json=test_data,
                auth=("testuser", "testpass")
            )

            assert response.status_code == 200
            # Verify that logging was called
            mock_to_csv.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
