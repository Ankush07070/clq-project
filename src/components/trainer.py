import sys

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging


def train_one_epoch(model, loader, optimizer, device, criterion=None, scaler=None):
    try:
        model.train()
        criterion = criterion or nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds, all_labels = [], []

        use_amp = scaler is not None and device.type == "cuda"
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        return {
            "loss": total_loss / max(1, len(loader)),
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }
    except Exception as e:
        logging.exception("Training failed")
        raise CustomException(e, sys)


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    try:
        model.eval()
        criterion = criterion or nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            "loss": total_loss / max(1, len(loader)),
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "predictions": all_preds,
            "labels": all_labels,
        }
    except Exception as e:
        logging.exception("Validation failed")
        raise CustomException(e, sys)
