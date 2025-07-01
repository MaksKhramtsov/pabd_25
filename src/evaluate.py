# evaluate.py
import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("lifecycle")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("lifecycle.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

processed_data_path = "./data/processed"
X_cols = ["total_meters", "floor", "floors_count", "rooms_count"]
y_cols = ["price"]


def evaluate_model(model_name):
    """Evaluate model performance"""
    logger.info("evaluating model")
    model = joblib.load(model_name)
    data = pd.read_csv(f"{processed_data_path}/test_data.csv")
    x_test, y_test = data[X_cols], data[y_cols]

    y_pred = model.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    logger.info(f"MAE: {metrics['mae']:.2f}")
    logger.info(f"MSE: {metrics['mse']:.2f}")
    logger.info(f"RMSE: {metrics['rmse']:.2f}")
    logger.info(f"R²: {metrics['r2']:.6f}")

    pd.DataFrame(metrics, index=[0]).to_json("metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=False, default="./models/xgboost_v1.pkl", help="Path to trained model")
    args = parser.parse_args()
    evaluate_model(args.model)
