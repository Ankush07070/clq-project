import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.exception import CustomException


LABEL_MAP = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}


@dataclass
class DataTransformation:
    """Prepare leakage-safe train/validation/test splits.

    Important: no oversampling is performed here. The training sampler is
    created only from the training split, so duplicated samples can never
    leak into validation/test data.
    """

    def encode_labels(self, df):
        try:
            df = df.copy()
            df["label"] = df["dx"].map(LABEL_MAP)
            if df["label"].isna().any():
                unknown = sorted(df.loc[df["label"].isna(), "dx"].dropna().unique())
                raise ValueError(f"Unknown diagnosis labels: {unknown}")
            df["label"] = df["label"].astype(int)
            return df, LABEL_MAP.copy()
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Group split by lesion_id to avoid the same lesion appearing in multiple splits."""
        try:
            df = df.reset_index(drop=True).copy()
            groups = df["lesion_id"] if "lesion_id" in df.columns else df["image_id"]

            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_val_idx, test_idx = next(gss_test.split(df, df["label"], groups=groups))
            train_val = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Convert the requested validation fraction to a fraction of the remaining data.
            relative_val = val_size / (1.0 - test_size)
            train_groups = (
                train_val["lesion_id"] if "lesion_id" in train_val.columns else train_val["image_id"]
            )
            gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=random_state)
            train_idx, val_idx = next(
                gss_val.split(train_val, train_val["label"], groups=train_groups)
            )

            train_df = train_val.iloc[train_idx].reset_index(drop=True)
            val_df = train_val.iloc[val_idx].reset_index(drop=True)
            return train_df, val_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def get_sample_weights(self, df):
        """Return per-row weights for a WeightedRandomSampler on TRAIN only."""
        try:
            counts = df["label"].value_counts().to_dict()
            return df["label"].map({label: 1.0 / count for label, count in counts.items()}).to_numpy()
        except Exception as e:
            raise CustomException(e, sys)

    def get_class_weights(self, df):
        """Return inverse-frequency class weights for optional weighted loss."""
        try:
            counts = df["label"].value_counts().sort_index()
            weights = len(df) / (len(counts) * counts)
            return weights.reindex(range(len(LABEL_MAP)), fill_value=0).to_numpy(dtype=np.float32)
        except Exception as e:
            raise CustomException(e, sys)
