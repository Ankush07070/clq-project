import torch

from src.components.model import SkinCancerModel


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    num_classes = int(config.get("num_classes", 7))
    model = SkinCancerModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config
