# House Price Prediction Application

A comprehensive machine learning application for predicting house prices. This project combines a FastAPI backend with a Streamlit frontend to provide an interactive house price prediction service.

## 🏠 Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard for house price predictions
- **RESTful API**: FastAPI backend with authentication for programmatic access
- **Machine Learning Model**: K-Nearest Neighbors regressor
- **Bulk Predictions**: Support for CSV file uploads for multiple predictions
- **Demographic Integration**: Incorporates zipcode demographic data for enhanced predictions
- **Model Monitoring**: Dashboard for tracking model performance and predictions
- **Docker Support**: Containerized deployment ready
- **MLflow Integration**: Model versioning and experiment tracking

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project-challenge
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   API_USERNAME=your_username
   API_PASSWORD=your_password
   EXPERIMENT_ID=your_experiment_id
   RUN_ID=your_run_id
   ```

### Running the Application

#### Option 1: Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t house-price-predictor .
   ```

2. **Run the API container**
   ```bash
   docker run -p 8000:8000 house-price-predictor
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

#### Option 2: Local Development

1. **Start the API server**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Usage

### Web Interface

1. **Manual Input Mode**
   - Enter property features: zipcode, bedrooms, bathrooms, square footage, etc.
   - Click "Predict" to get instant price predictions
   - View prediction results with confidence metrics

2. **Bulk Predictions**
   - Upload a CSV file with multiple property records
   - Expected columns: `zipcode`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement`
   - Get predictions for all properties in the file

### API Usage

#### Authentication
The API uses HTTP Basic Authentication. Include credentials in your requests:

```python
import requests
from requests.auth import HTTPBasicAuth

response = requests.post(
    "http://localhost:8000/predict",
    json=input_data,
    auth=HTTPBasicAuth("username", "password")
)
```

#### Single Prediction
```python
import requests

data = {
    "zipcode": 98042,
    "bedrooms": 4.0,
    "bathrooms": 1.0,
    "sqft_living": 1680.0,
    "sqft_lot": 5043.0,
    "floors": 1.5,
    "sqft_above": 1680.0,
    "sqft_basement": 1911.0
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
```

#### Bulk Predictions
```python
response = requests.post("http://localhost:8000/predict_full", json=data)
```
