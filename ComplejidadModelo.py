import torch
from torchvision import models
import torch.nn as nn
import os
import time


MODELO = "modelo_detector_grietas.pth"

IMG_SIZE = (224, 224)



model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    2
)

model.load_state_dict(
    torch.load(
        MODELO,
        map_location="cpu"
    )
)

model.eval()



parametros_totales = sum(
    p.numel()
    for p in model.parameters()
)

parametros_entrenables = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)



tamano_bytes = os.path.getsize(
    MODELO
)

tamano_mb = tamano_bytes / (
    1024 * 1024
)



imagen = torch.randn(
    1,
    3,
    224,
    224
)



with torch.no_grad():

    for _ in range(10):
        model(imagen)



tiempos = []

with torch.no_grad():

    for _ in range(100):

        inicio = time.perf_counter()

        model(imagen)

        fin = time.perf_counter()

        tiempos.append(
            (fin - inicio) * 1000
        )


tiempo_promedio = sum(tiempos) / len(
    tiempos
)

tiempo_minimo = min(tiempos)

tiempo_maximo = max(tiempos)



print("=" * 50)
print("COMPLEJIDAD DEL MODELO")
print("=" * 50)

print()

print(
    "Arquitectura: MobileNetV2"
)

print(
    "Entrada:",
    "224 x 224 RGB"
)

print()

print(
    "Parámetros totales:",
    f"{parametros_totales:,}"
)

print(
    "Parámetros entrenables:",
    f"{parametros_entrenables:,}"
)

print()

print(
    "Tamaño del modelo:",
    f"{tamano_mb:.2f} MB"
)

print()

print(
    "Tiempo promedio de inferencia:",
    f"{tiempo_promedio:.2f} ms"
)

print(
    "Tiempo mínimo:",
    f"{tiempo_minimo:.2f} ms"
)

print(
    "Tiempo máximo:",
    f"{tiempo_maximo:.2f} ms"
)

print()

print("=" * 50)
print("EVALUACIÓN DE COMPLEJIDAD TERMINADA")
print("=" * 50)