import torch
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import numpy as np


DATASET = "dataset"
MODELO = "modelo_detector_grietas.pth"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(
    DATASET,
    transform=transform
)


total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size


generator = torch.Generator().manual_seed(SEED)

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator
)


test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("=" * 50)
print("EVALUACIÓN DEL MODELO")
print("=" * 50)

print("\nClases:")
print(dataset.classes)

print("\nImágenes del conjunto de prueba:")
print(len(test_ds))


model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(dataset.classes)
)


model.load_state_dict(
    torch.load(
        MODELO,
        map_location="cpu"
    )
)

model.eval()


device = torch.device("cpu")

model = model.to(device)


predicciones = []
etiquetas_reales = []


print("\nEvaluando modelo...")


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        predicciones.extend(
            predictions.cpu().numpy()
        )

        etiquetas_reales.extend(
            labels.numpy()
        )


predicciones = np.array(
    predicciones
)

etiquetas_reales = np.array(
    etiquetas_reales
)


accuracy = accuracy_score(
    etiquetas_reales,
    predicciones
)

precision = precision_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    etiquetas_reales,
    predicciones,
    average="binary",
    pos_label=1,
    zero_division=0
)


correctas = np.sum(
    etiquetas_reales == predicciones
)

incorrectas = np.sum(
    etiquetas_reales != predicciones
)


matriz = confusion_matrix(
    etiquetas_reales,
    predicciones
)


print("\n" + "=" * 50)
print("RESULTADOS")
print("=" * 50)

print(
    f"\nImágenes evaluadas: {len(test_ds)}"
)

print(
    f"Correctas: {correctas}"
)

print(
    f"Incorrectas: {incorrectas}"
)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall: {recall * 100:.2f}%"
)

print(
    f"F1-score: {f1 * 100:.2f}%"
)


print("\nMatriz de confusión:")

print(matriz)


print("\nReporte de clasificación:")

print(
    classification_report(
        etiquetas_reales,
        predicciones,
        target_names=dataset.classes,
        zero_division=0
    )
)


print("\n" + "=" * 50)
print("Evaluación terminada.")
print("=" * 50)