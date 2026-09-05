import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from src.components.data_ingestion import DataIngestion
from src.components.data_loader import create_dataloader, create_weighted_sampler
from src.components.data_transformation import DataTransformation, LABEL_MAP
from src.components.data_validation import DataValidation
from src.components.model import SkinCancerModel
from src.components.trainer import train_one_epoch, validate
from src.components.transforms import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_train_transforms, get_val_transforms
from src.components.utils import save_checkpoint, save_json
from src.exception import CustomException
from src.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.csv_path = os.getenv("METADATA_CSV", "data/raw/HAM10000_metadata.csv")
        self.image_dir = os.getenv("IMAGE_DIR", "data/raw/images")
        self.save_dir = "data/processed"
        self.artifact_dir = "artifacts"
        self.batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self.epochs = int(os.getenv("EPOCHS", "15"))
        self.lr = float(os.getenv("LEARNING_RATE", "0.0001"))
        self.patience = int(os.getenv("EARLY_STOPPING_PATIENCE", "4"))

    @staticmethod
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_pipeline(self):
        try:
            self.seed_everything()
            logging.info("Starting leakage-safe training pipeline")

            ingestion = DataIngestion(self.csv_path, self.image_dir)
            df = ingestion.add_image_paths(ingestion.load_data())

            validation = DataValidation()
            df = validation.validate(df)

            transform = DataTransformation()
            df, label_map = transform.encode_labels(df)
            train_df, val_df, test_df = transform.split_data(df)

            # Save clean, non-overlapping splits. No oversampling is written to disk.
            os.makedirs(self.save_dir, exist_ok=True)
            train_df.to_csv(os.path.join(self.save_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(self.save_dir, "val.csv"), index=False)
            test_df.to_csv(os.path.join(self.save_dir, "test.csv"), index=False)

            # Weighted sampler is built ONLY from training data.
            sampler = create_weighted_sampler(train_df)
            train_loader = create_dataloader(
                train_df, get_train_transforms(), self.batch_size, shuffle=False, sampler=sampler
            )
            val_loader = create_dataloader(
                val_df, get_val_transforms(), self.batch_size, shuffle=False
            )
            test_loader = create_dataloader(
                test_df, get_val_transforms(), self.batch_size, shuffle=False
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SkinCancerModel(num_classes=len(label_map), pretrained=True).to(device)

            # Freeze the first phase of the pretrained backbone for stable fine-tuning.
            for parameter in model.cnn.parameters():
                parameter.requires_grad = False

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )
            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

            config = {
                "architecture": "MobileNetV2 + Transformer",
                "num_classes": len(label_map),
                "label_map": label_map,
                "image_size": IMAGE_SIZE,
                "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "seed": 42,
                "split_strategy": "GroupShuffleSplit by lesion_id",
                "train_sampler": "WeightedRandomSampler on train only",
            }
            save_json(config, os.path.join(self.artifact_dir, "model_config.json"))

            best_f1 = -1.0
            epochs_without_improvement = 0
            history = []

            for epoch in range(self.epochs):
                train_metrics = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
                val_metrics = validate(model, val_loader, device, criterion)
                scheduler.step(val_metrics["macro_f1"])

                record = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                }
                history.append(record)
                logging.info("Epoch %s: %s", epoch + 1, json.dumps(record))

                if val_metrics["macro_f1"] > best_f1:
                    best_f1 = val_metrics["macro_f1"]
                    epochs_without_improvement = 0
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        {k: v for k, v in val_metrics.items() if k not in {"predictions", "labels"}},
                        os.path.join(self.artifact_dir, "best_model.pth"),
                        config,
                    )
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        logging.info("Early stopping at epoch %s", epoch + 1)
                        break

                # Unfreeze the last MobileNet stage after the first epoch for fine-tuning.
                if epoch == 0:
                    for parameter in model.cnn[-3:].parameters():
                        parameter.requires_grad = True
                    optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr * 0.1, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max", factor=0.5, patience=2
                    )

            save_json(history, os.path.join(self.artifact_dir, "training_history.json"))

            # Load the best checkpoint before final test evaluation.
            checkpoint = torch.load(os.path.join(self.artifact_dir, "best_model.pth"), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_metrics = validate(model, test_loader, device, criterion)
            report = classification_report(
                test_metrics["labels"],
                test_metrics["predictions"],
                target_names=list(label_map.keys()),
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"]).tolist()

            final_metrics = {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_precision_macro": test_metrics["precision_macro"],
                "test_recall_macro": test_metrics["recall_macro"],
                "classification_report": report,
                "confusion_matrix": cm,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "unique_train_images": train_df["image_id"].nunique(),
                "unique_val_images": val_df["image_id"].nunique(),
                "unique_test_images": test_df["image_id"].nunique(),
            }
            save_json(final_metrics, os.path.join(self.artifact_dir, "test_metrics.json"))
            logging.info("Final test metrics: %s", json.dumps({k: v for k, v in final_metrics.items() if k != "classification_report"}))

            print("\nTraining complete.")
            print(f"Best validation macro F1: {best_f1:.4f}")
            print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"Artifacts: {self.artifact_dir}/best_model.pth")

        except Exception as e:
            logging.exception("Training pipeline failed")
            raise CustomException(e, sys)
