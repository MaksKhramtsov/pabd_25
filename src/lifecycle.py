import argparse
import datetime
import glob

import joblib
import numpy as np
import cianparser
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

import logging
from sklearn.model_selection import train_test_split
logger = logging.getLogger('lifecycle')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('lifecycle.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

TRAIN_SIZE = 0.2
MODEL_NAME = "./models/catboost_v1.pkl"

raw_data_path = './data/raw'
processed_data_path = './data/processed'
X_cols = ['total_meters', 'floor', 'floors_count', 'rooms_count']
y_cols = ['price']


def parse_cian():
    """Parse data to data/raw"""
    logger.info("parsing cian apartments")
    moscow_parser = cianparser.CianParser(location="Москва")
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = (1, 2, 3, 'studio')
    csv_path = f'{raw_data_path}/{t}_{'_'.join(map(str, n_rooms))}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=n_rooms,
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


def preprocess_data():
    """Filter and remove"""
    logger.info('preprocessing data')

    file_list = glob.glob(raw_data_path + "/*.csv")
    logger.info(f"found files: {file_list}")
    main_dataframe = pd.read_csv(file_list[0], delimiter=',')
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    data = main_dataframe[['url_id', 'total_meters', 'price', 'floor', 'floors_count', 'rooms_count']].set_index(
        'url_id')
    data = data[data['price'] < 100_000_000]
    data.sort_values('url_id', inplace=True)
    data.to_csv(f"{processed_data_path}/train_data.csv")


def train_model(split_size, model_name):
    """Train model and save with MODEL_NAME"""
    logger.info('training model')
    data = pd.read_csv(f"{processed_data_path}/train_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
    data[X_cols],
    data[y_cols],
    test_size = split_size,
    shuffle = False
    )

    train_pool = Pool(
        data=X_train,
        label=y_train,
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
    )

    params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'early_stopping_rounds': 50,
        'verbose': 100
    }

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True
    )

    joblib.dump(model, model_name)
    test_data = X_test.copy()
    test_data['price'] = y_test
    test_data.to_csv(f"{processed_data_path}/test_data.csv")


def test_model(model_name):
    """Test model with new data"""
    logger.info('testing model')
    model = joblib.load(model_name)
    data = pd.read_csv(f"{processed_data_path}/test_data.csv")
    x_test, y_test = data[X_cols], data[y_cols]

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # Вывод метрик качества
    logger.info(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    logger.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    logger.info(f"Корень из среднеквадратичной ошибки (RMSE): {rmse:.2f}")
    logger.info(f"Коэффициент детерминации R²: {r2:.6f}")



if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    split = args.split
    model_path = args.model
    logger.info(f'launched with params: split={split}, model_path={model_path}')

    # parse_cian()
    preprocess_data()
    train_model(split, model_path)
    test_model(model_path)