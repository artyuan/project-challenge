import json
import pathlib
import pickle
import mlflow
from typing import List
from typing import Tuple
import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved
run_name = f"KNN_HousePrice_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv("data/zipcode_demographics.csv",
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    scaler = preprocessing.RobustScaler()
    regressor = neighbors.KNeighborsRegressor()

    mlflow.set_experiment("House_Pricemlflow server --host 127.0.0.1 --port 8080")

    with mlflow.start_run(run_name=run_name):
        # Log model parameters
        mlflow.log_param("model_type", "KNeighborsRegressor")
        mlflow.log_param("n_neighbors", regressor.n_neighbors)
        mlflow.log_param("scaler", scaler.__class__.__name__)

        # Train model
        model = pipeline.make_pipeline(scaler, regressor).fit(x_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(_x_test)
        rmse = metrics.mean_squared_error(_y_test, y_pred, squared=False)
        r2 = metrics.r2_score(_y_test, y_pred)
        mae = metrics.mean_absolute_error(_y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save artifacts
        output_dir = pathlib.Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        pickle_path = output_dir / "model.pkl"
        json_path = output_dir / "model_features.json"

        pickle.dump(model, open(pickle_path, 'wb'))
        json.dump(list(x_train.columns), open(json_path, 'w'))

        mlflow.log_artifact(pickle_path.as_posix())
        mlflow.log_artifact(json_path.as_posix())

        # Log model with MLflow model registry (optional)
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
