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

El dataset original (`dataset.zip.rar`) **no está incluido en este repositorio** porque pesa aproximadamente **231 MiB** y GitHub bloquea las subidas normales de archivos individuales de más de **100 MB**. No se pudo subir directamente por ese límite de tamaño; no es un error del código del proyecto.

Para usar el entrenamiento, descarga el dataset desde la ubicación original donde se haya compartido, extráelo y coloca las imágenes con esta estructura:

```text
dataset/
├── Negative/
└── Positive/
```

Los nombres de las carpetas deben conservarse exactamente. Consulta [dataset/README.md](dataset/README.md) para ver los pasos de preparación. Si se quiere distribuir el dataset desde GitHub, debe publicarse mediante Git LFS (revisando sus cuotas) o alojarse en un servicio externo y enlazarse aquí.

**Nota sobre el modelo:** el archivo binario `modelo_detector_grietas.pth` tampoco se encuentra incluido en el repositorio actualmente. Para ejecutar la interfaz sin reentrenar, coloca ese archivo en la raíz del proyecto. Si no tienes los pesos entrenados, será necesario obtenerlos desde el equipo donde se guardaron o volver a entrenar cuando el dataset esté disponible.

## Datos y modelo

El archivo original del dataset se llama `dataset.zip.rar` y pesa aproximadamente 231 MiB. GitHub bloquea archivos individuales de más de 100 MB en las subidas normales, por lo que **el archivo comprimido completo no se puede agregar directamente al repositorio mediante una carga normal**. Consulta [dataset/README.md](dataset/README.md) para preparar las carpetas y las opciones para alojar el archivo grande con Git LFS o almacenamiento externo.

El modelo entrenado `modelo_detector_grietas.pth` también es un archivo binario. Para usar la interfaz, debe estar en la raíz del proyecto junto a `PruebaDetector.py`. Si no está disponible en la descarga, cópialo desde la carpeta donde se entrenó.

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
