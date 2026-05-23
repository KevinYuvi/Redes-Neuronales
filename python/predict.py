import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=True
)

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

model.load_state_dict(
    torch.load("mnist_model.pth", map_location=device)
)

model.eval()

images, labels = next(iter(test_loader))

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

image = images[0].squeeze().numpy()

plt.imshow(image, cmap="gray")

plt.title(
    f"Real: {labels.item()} | Prediccion IA: {predicted.item()}"
)

plt.axis("off")

plt.savefig("prediccion.png")

print("Imagen guardada como prediccion.png")
print("Numero real:", labels.item())
print("Prediccion IA:", predicted.item())