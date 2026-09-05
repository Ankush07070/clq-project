import os
import sys
from dataclasses import dataclass

from PIL import Image

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataValidation:
    """Validate paths and image readability before model training."""

    def validate(self, df):
        try:
            required = {"image_id", "dx", "image_path"}
            missing = required.difference(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {sorted(missing)}")

            valid_rows = []
            corrupt = 0
            missing_files = 0
            for _, row in df.iterrows():
                path = row["image_path"]
                if not os.path.isfile(path):
                    missing_files += 1
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    valid_rows.append(True)
                except Exception:
                    corrupt += 1
                    valid_rows.append(False)

            if len(valid_rows) != len(df):
                # The loop above appends only for existing files, so rebuild cleanly below.
                clean_mask = []
                for _, row in df.iterrows():
                    path = row["image_path"]
                    if not os.path.isfile(path):
                        clean_mask.append(False)
                        continue
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        clean_mask.append(True)
                    except Exception:
                        clean_mask.append(False)
                df = df.loc[clean_mask].copy()
            else:
                df = df.copy()

            logging.info("Validation: missing=%s corrupt=%s remaining=%s", missing_files, corrupt, len(df))
            return df.reset_index(drop=True)
        except Exception as e:
            logging.exception("Data validation failed")
            raise CustomException(e, sys)

    def remove_missing_images(self, df):
        # Backward-compatible alias used by older code.
        return self.validate(df)
