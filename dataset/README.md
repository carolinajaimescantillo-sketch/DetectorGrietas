# Dataset de DetectorGrietas

## Enlace de descarga

El dataset de referencia seleccionado para este proyecto es **Surface Crack Detection** en Kaggle:

**[Abrir y descargar Surface Crack Detection en Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)**

Kaggle puede pedir que inicies sesión y aceptes sus condiciones de uso. Este enlace lleva a la página del dataset, no directamente al archivo local `dataset.zip.rar` utilizado durante el desarrollo.

## Estructura requerida

Extrae los datos de modo que las imágenes queden organizadas así:

```text
dataset/
├── Negative/   # Imágenes de superficies sin grietas
└── Positive/   # Imágenes de superficies con grietas
```

Los nombres de las carpetas son importantes porque el entrenamiento usa `torchvision.datasets.ImageFolder`.

## Preparar los datos en Windows

1. Abre el enlace de Kaggle de arriba y descarga el dataset.
2. Extrae el archivo comprimido con 7-Zip o WinRAR.
3. Organiza las imágenes en las carpetas `Negative` y `Positive` dentro de `dataset`.
4. Comprueba que no haya un nivel extra de carpetas. La ruta final debe ser, por ejemplo, `dataset/Negative/imagen.jpg`, no `dataset/otra_carpeta/Negative/imagen.jpg`.
5. Ejecuta `python DetectorGrietas.py` solo si quieres volver a entrenar.

## Por qué el archivo local no está guardado aquí

El archivo `dataset.zip.rar` utilizado durante el desarrollo pesa aproximadamente 231 MiB. GitHub bloquea archivos individuales de más de 100 MB en las subidas normales. Para distribuir ese archivo comprimido exacto desde el repositorio, hay que usar **Git LFS** (y comprobar la cuota de almacenamiento/ancho de banda) o publicarlo en un servicio externo y enlazarlo desde el README.

Este repositorio incluye el enlace a la fuente del dataset y las instrucciones de preparación, pero no las imágenes del dataset en sí.
