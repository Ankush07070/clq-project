import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestion:
    csv_path: str
    image_dir: str

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                raise ValueError("Metadata CSV is empty")
            logging.info("Loaded metadata: %s rows", len(df))
            return df
        except Exception as e:
            logging.exception("Data ingestion failed")
            raise CustomException(e, sys)

    def add_image_paths(self, df):
        try:
            df = df.copy()
            df["image_path"] = df["image_id"].astype(str).map(
                lambda x: os.path.join(self.image_dir, f"{x}.jpg")
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)
