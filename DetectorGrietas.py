import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torchvision import models


DATASET = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 20


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


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nInformación del dataset")

print("Clases:")
print(dataset.classes)

print("\nCantidad total de imágenes:")
print(len(dataset))

print("\nDistribución:")
print("Entrenamiento → 70 %")
print("Validación    → 15 %")
print("Prueba        → 15 %")

print("\nCantidad de imágenes:")
print("Entrenamiento →", len(train_ds))
print("Validación    →", len(val_ds))
print("Prueba        →", len(test_ds))


model = models.mobilenet_v2(
    weights="DEFAULT"
)


for param in model.parameters():
    param.requires_grad = False


num_classes = len(dataset.classes)

model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)


print("\nModelo preparado.")
print("Modelo: MobileNetV2")
print("Dispositivo utilizado:", device)
print("Número de clases:", num_classes)


print("\nIniciando entrenamiento...")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = correct / total


    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = val_correct / val_total


    print(
        f"Época {epoch + 1}/{EPOCHS} "
        f"- pérdida: {running_loss / len(train_loader):.4f} "
        f"- accuracy: {train_accuracy * 100:.2f}% "
        f"- val_accuracy: {val_accuracy * 100:.2f}%"
    )


torch.save(
    model.state_dict(),
    "modelo_detector_grietas.pth"
)

print("\nModelo guardado.")


model.eval()

correctas = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correctas += (
            predictions == labels
        ).sum().item()


incorrectas = total - correctas

accuracy_test = correctas / total


print("\nEvaluación final del modelo")

print(
    "Imágenes de prueba:",
    total
)

print(
    "Predicciones correctas:",
    correctas
)

print(
    "Predicciones incorrectas:",
    incorrectas
)

print(
    "Accuracy en prueba:",
    round(accuracy_test * 100, 2),
    "%"
)

print("\nModelo guardado correctamente.")
