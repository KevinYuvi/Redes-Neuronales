from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_model.pth"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def preprocess_image(image: Image.Image):
    image = image.convert("L")

    # Asegurar fondo negro y trazo blanco
    image = ImageOps.invert(image) if image.getpixel((0, 0)) > 128 else image

    # Recortar bordes vacíos
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Mantener proporción y meter en 20x20
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Crear canvas MNIST 28x28
    new_image = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    new_image.paste(image, (x, y))

    tensor = transforms.ToTensor()(new_image)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

    return tensor.unsqueeze(0)

def predict_python(image: Image.Image):
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "language": "Python",
        "prediction": int(predicted.item()),
        "confidence": round(float(confidence.item() * 100), 2)
    }