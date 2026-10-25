# Detector de Grietas

Sistema académico para la clasificación preliminar de superficies con o sin grietas mediante **MobileNetV2, PyTorch y OpenCV**. Incluye una interfaz gráfica de escritorio con Tkinter y scripts para entrenar, probar y evaluar el modelo.

> **Aviso de seguridad:** este proyecto es una ayuda visual académica. No certifica la seguridad estructural, no determina la resistencia de una edificación ni predice terremotos. Ante grietas diagonales, ramificadas o que aumenten, deformaciones, desprendimientos, filtraciones o ruidos, solicita una inspección profesional aunque el resultado sea bajo.

## Contenido del repositorio

- `DetectorGrietas.py`: entrenamiento de MobileNetV2.
- `PruebaDetector.py`: interfaz gráfica de escritorio.
- `EvaluarModelo.py`: métricas de clasificación y matriz de confusión.
- `EvaluarTodo.py`: evaluación adicional.
- `ComplejidadModelo.py`: parámetros, tamaño de pesos y tiempo de inferencia.
- `LocalizarGrieta.py`: procesamiento visual aproximado mediante OpenCV.
- `MedirGrieta.py`: estimaciones en píxeles.
- `RiesgoGrieta.py`: semáforo de riesgo orientativo.
- `dataset/README.md`: instrucciones para preparar los datos.
- `modelo_detector_grietas.pth`: pesos entrenados necesarios para hacer predicciones (se deben descargar/copiar por separado mientras no estén publicados como artefacto binario).

## Importante: dataset no incluido en GitHub

El dataset de referencia seleccionado para este proyecto es **Surface Crack Detection** de Kaggle: [abrir y descargar el dataset en Kaggle](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection). Kaggle puede solicitar iniciar sesión y aceptar sus condiciones de uso.

El archivo local `dataset.zip.rar` utilizado durante el desarrollo pesa aproximadamente **231 MiB** y no está incluido en este repositorio porque GitHub bloquea las subidas normales de archivos individuales de más de **100 MB**. El enlace anterior lleva a la página de la fuente original; no es un enlace directo al archivo local `dataset.zip.rar`.

Después de descargar los datos, extráelos y organiza las imágenes con esta estructura:

```text
dataset/
├── Negative/
└── Positive/
```

Conserva exactamente los nombres de las carpetas. Consulta [dataset/README.md](dataset/README.md) para ver los pasos de preparación. Para distribuir el archivo local comprimido exacto, habría que publicarlo mediante Git LFS (revisando sus cuotas) o alojarlo en un servicio externo y enlazarlo aquí.

## Descargar el proyecto

1. Abre https://github.com/carolinajaimescantillo-sketch/DetectorGrietas.
2. Pulsa **Code → Download ZIP**.
3. Descomprime el proyecto.
4. Prepara el dataset siguiendo [dataset/README.md](dataset/README.md).
5. Copia `modelo_detector_grietas.pth` en la carpeta raíz para ejecutar predicciones sin reentrenar.

También puedes clonar el repositorio:

```powershell
git clone https://github.com/carolinajaimescantillo-sketch/DetectorGrietas.git
cd DetectorGrietas
code .
```

## Instalación en Windows

Se recomienda Python 3.11 o 3.12. Abre PowerShell dentro de la carpeta del proyecto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si no puedes activar el entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En VS Code selecciona el intérprete del entorno virtual con **Ctrl+Shift+P → Python: Select Interpreter → .venv**.

## Ejecutar la interfaz

Asegúrate de tener `modelo_detector_grietas.pth` en la raíz y ejecuta:

```powershell
python PruebaDetector.py
```

## Entrenar el modelo

La estructura de datos esperada es:

```text
DetectorGrietas/
├── dataset/
│   ├── Negative/
│   └── Positive/
├── modelo_detector_grietas.pth
├── DetectorGrietas.py
└── PruebaDetector.py
```

- `Negative`: imágenes de superficies sin grietas.
- `Positive`: imágenes de superficies con grietas.

Cuando las carpetas estén listas, ejecuta:

```powershell
python DetectorGrietas.py
```

El entrenamiento separa los datos en entrenamiento (70 %), validación (15 %) y prueba (15 %), usa semilla 123 y guarda los pesos en `modelo_detector_grietas.pth`. Entrenar nuevamente no es necesario si ya tienes los pesos entrenados.

## Evaluar el modelo

Con el dataset y los pesos disponibles:

```powershell
python EvaluarModelo.py
python ComplejidadModelo.py
```

La evaluación informa Accuracy, Precision, Recall, F1-score y matriz de confusión. La complejidad calcula el número de parámetros, el tamaño de los pesos y el tiempo de inferencia en el equipo actual.

## Limitaciones

- El modelo clasifica imágenes en las clases aprendidas `Negative` y `Positive`.
- La confianza no es una probabilidad certificada de daño estructural.
- OpenCV puede resaltar texturas, juntas, sombras u objetos que no sean grietas.
- Las medidas son píxeles, no milímetros, a menos que se calibre con una referencia física.
- Un resultado `Negative` no garantiza que una estructura sea segura.

## Tecnologías

Python · PyTorch · torchvision · MobileNetV2 · OpenCV · NumPy · Pillow · scikit-learn · Tkinter
