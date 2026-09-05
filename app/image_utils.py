from PIL import Image
from torchvision import transforms

from src.components.transforms import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def process_image(image: Image.Image):
    """Inference preprocessing must exactly match validation preprocessing."""
    return INFERENCE_TRANSFORM(image).unsqueeze(0)
