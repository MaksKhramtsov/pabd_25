import os
from dataclasses import dataclass
from logging.config import dictConfig

import boto3
import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

LOCAL_MODEL_PATH = "../models/catboost_v1.pkl"

load_dotenv()

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)
CORS(app)


auth = HTTPTokenAuth(scheme="Bearer")
tokens = [os.getenv("API_TOKEN")]


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return True
    return False


@dataclass
class HouseInfo:
    area: int = None
    rooms: int = None
    total_floors: int = None
    floor: int = None


def load_model_from_s3():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ru-central1"),
        )

        s3.download_file("pabd25", "khramtsov/models/catboost_v1.pkl", LOCAL_MODEL_PATH)
        return joblib.load(LOCAL_MODEL_PATH)
    except Exception as e:
        app.logger.error(f"Ошибка загрузки модели из S3: {e}")
        return None


if os.getenv("FLASK_ENV") == "production":
    loaded_model = load_model_from_s3()
    if loaded_model is None:
        raise RuntimeError("Не удалось загрузить модель из S3 в production режиме")
else:
    try:
        loaded_model = joblib.load(LOCAL_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить локальную модель: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/numbers", methods=["POST"])
@auth.login_required
def process_numbers():
    data = request.get_json()
    app.logger.info(f"Получены данные: {data}")

    try:
        input_data = {
            "total_meters": [float(data["area"])],
            "floor": [int(data["floor"])],
            "floors_count": [int(data["total_floors"])],
            "rooms_count": [int(data["rooms"])],
        }

        input_df = pd.DataFrame(input_data)
        predicted = loaded_model.predict(input_df)
        price = int(predicted[0])

        return jsonify({"status": "success", "data": price, "currency": "RUB"})

    except ValueError as e:
        app.logger.error(f"Ошибка валидации: {e}")
        return jsonify({"status": "error", "message": "Некорректные входные данные"}), 400
    except Exception as e:
        app.logger.error(f"Неожиданная ошибка: {e}")
        return jsonify({"status": "error", "message": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "False") == "True",
    )
