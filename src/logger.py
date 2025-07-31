import os
import pandas as pd
from datetime import datetime
from uuid import uuid4
from src.config import settings

LOG_FILE = "data/prediction_logs.csv"
LOG_FILE_ALL = "data/prediction_logs_all_inputs.csv"

def log_prediction(input_data: dict, full: bool = False):
    log = {
        'id': input_data['id'],
        'timestamp': input_data['timestamp'],
        'prediction': input_data['prediction'][0],
        'experiment_id': input_data['model']['experiment_id'],
        'run_id': input_data['model']['run_id'],
        **input_data['features']
    }

    df = pd.DataFrame([log])
    log_path = LOG_FILE_ALL if full else LOG_FILE

    if not os.path.isfile(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)

