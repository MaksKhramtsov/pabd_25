# preprocess.py
import glob
import logging

import pandas as pd

logger = logging.getLogger("lifecycle")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("lifecycle.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

raw_data_path = "./data/raw"
processed_data_path = "./data/processed"


def preprocess_data():
    """Filter and clean data"""
    logger.info("preprocessing data")
    file_list = glob.glob(raw_data_path + "/*.csv")
    logger.info(f"found files: {file_list}")

    main_dataframe = pd.read_csv(file_list[0], delimiter=",")
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i], delimiter=",")
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe["url_id"] = main_dataframe["url"].map(lambda x: x.split("/")[-2])
    data = main_dataframe[["url_id", "total_meters", "price", "floor", "floors_count", "rooms_count"]].set_index(
        "url_id"
    )
    data = data[data["price"] < 100_000_000]
    data.sort_values("url_id", inplace=True)
    data.to_csv(f"{processed_data_path}/train_data.csv")


if __name__ == "__main__":
    preprocess_data()
