import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.exception import CustomException


class SkinCancerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
            return image, torch.tensor(int(row["label"]), dtype=torch.long)
        except Exception as e:
            raise CustomException(e, sys)


def create_dataloader(df, transform, batch_size=32, shuffle=True, sampler=None, num_workers=2):
    try:
        dataset = SkinCancerDataset(df, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        return loader
    except Exception as e:
        raise CustomException(e, sys)


def create_weighted_sampler(df):
    try:
        counts = df["label"].value_counts().to_dict()
        weights = df["label"].map({label: 1.0 / count for label, count in counts.items()})
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights.to_numpy(), dtype=torch.double),
            num_samples=len(df),
            replacement=True,
        )
    except Exception as e:
        raise CustomException(e, sys)
