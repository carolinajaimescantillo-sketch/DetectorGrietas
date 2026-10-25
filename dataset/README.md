# Dataset de DetectorGrietas

## Estructura requerida

Extrae el archivo de datos de modo que las imágenes queden organizadas así:

```text
dataset/
├── Negative/   # Imágenes de superficies sin grietas
└── Positive/   # Imágenes de superficies con grietas
```

Los nombres de las carpetas son importantes porque el entrenamiento usa `torchvision.datasets.ImageFolder`.

## Preparar los datos en Windows

1. Descarga el archivo comprimido original `dataset.zip.rar` desde la ubicación donde se comparta.
2. Extrae el contenido con 7-Zip o WinRAR.
3. Si el archivo extraído contiene directamente las carpetas `Negative` y `Positive`, colócalas dentro de una carpeta llamada `dataset`.
4. Comprueba que no haya un nivel extra de carpetas. La ruta final debe ser, por ejemplo, `dataset/Negative/imagen.jpg`, no `dataset/otra_carpeta/Negative/imagen.jpg`.
5. Ejecuta `python DetectorGrietas.py` solo si quieres volver a entrenar.

## Por qué el archivo no está guardado aquí

El archivo `dataset.zip.rar` compartido para este proyecto pesa aproximadamente 231 MiB. GitHub bloquea archivos individuales de más de 100 MB en las subidas normales. Para distribuirlo desde GitHub hay que usar **Git LFS** (y comprobar la cuota de almacenamiento/ancho de banda de la cuenta) o publicar el archivo en un servicio de almacenamiento externo y enlazarlo desde el README.

Este documento deja registrada la estructura y el procedimiento, pero **no contiene las imágenes del dataset** ni un enlace de descarga inventado.
