import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

dataset = datasets.ImageFolder(
    "dataset",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("modelo_detector_grietas.pth", map_location="cpu"))
model.eval()

correctas = 0
total = 0

with torch.no_grad():
    for imagenes, etiquetas in loader:
        predicciones = model(imagenes).argmax(1)
        correctas += (predicciones == etiquetas).sum().item()
        total += etiquetas.size(0)

print("Imágenes:", total)
print("Correctas:", correctas)
print("Incorrectas:", total - correctas)
print("Accuracy:", round(correctas / total * 100, 2), "%")