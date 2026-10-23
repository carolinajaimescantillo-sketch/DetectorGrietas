# Detector de Grietas

Sistema académico para la clasificación preliminar de superficies con o sin grietas mediante **MobileNetV2, PyTorch y OpenCV**. Incluye una interfaz gráfica de escritorio con Tkinter y los scripts para entrenar, probar y evaluar el modelo.

> **Importante:** es una herramienta de evaluación visual preliminar. No certifica la seguridad estructural, no determina la resistencia de una edificación ni predice terremotos. Ante grietas diagonales, ramificadas o que aumenten, deformaciones, desprendimientos, filtraciones o ruidos, solicita una inspección profesional incluso si el resultado del programa es bajo.

## Qué incluye

- `PruebaDetector.py`: interfaz gráfica de escritorio para seleccionar una imagen y ver el resultado del modelo.
- `DetectorGrietas.py`: entrenamiento de MobileNetV2 durante 20 épocas.
- `EvaluarModelo.py`: métricas de clasificación y matriz de confusión.
- `EvaluarTodo.py`: evaluación adicional del conjunto de imágenes.
- `ComplejidadModelo.py`: parámetros, tamaño del modelo y tiempo de inferencia.
- `LocalizarGrieta.py`: procesamiento de imágenes, máscara visual y funciones auxiliares.
- `MedirGrieta.py`: estimaciones en píxeles de la zona candidata.
- `RiesgoGrieta.py`: ventana independiente del semáforo de riesgo.

No se incluye una aplicación web. La interfaz principal es de escritorio y el código se puede abrir y editar en Visual Studio Code.

## Requisitos

- Windows, macOS o Linux.
- Python 3.11 o 3.12 recomendado.
- Visual Studio Code (opcional, para editar el código).
- El archivo de pesos `modelo_detector_grietas.pth` para usar la interfaz y ejecutar las evaluaciones.
- Las carpetas de imágenes `dataset/Negative` y `dataset/Positive` para entrenar o volver a evaluar el modelo.

## Descargar el proyecto

1. Abre el repositorio: https://github.com/carolinajaimescantillo-sketch/DetectorGrietas
2. Pulsa **Code → Download ZIP**.
3. Descomprime el ZIP en una carpeta, por ejemplo `C:\Proyecto_algoritmos`.
4. Abre esa carpeta en Visual Studio Code con **File → Open Folder**.

También puedes clonar el repositorio desde una terminal:

```powershell
git clone https://github.com/carolinajaimescantillo-sketch/DetectorGrietas.git
cd DetectorGrietas
code .
```

## Preparar el entorno en Windows

Abre la terminal integrada de Visual Studio Code dentro de la carpeta del proyecto y ejecuta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell impide activar el entorno, puedes ejecutar los comandos directamente con la ruta del entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para evitar instalar paquetes en otro Python, comprueba que la terminal use el intérprete de `.venv`. En VS Code: **Ctrl+Shift+P → Python: Select Interpreter → .venv**.

## Usar la interfaz gráfica

1. Asegúrate de que `modelo_detector_grietas.pth` esté en la carpeta raíz, junto a `PruebaDetector.py`.
2. Activa el entorno virtual e instala las dependencias.
3. Ejecuta:

```powershell
python PruebaDetector.py
```

4. En la ventana, selecciona una imagen y ejecuta el análisis.

Si aparece un mensaje indicando que no se pudo cargar el modelo, revisa que el archivo `modelo_detector_grietas.pth` exista en la carpeta raíz. **El archivo de pesos no está incluido actualmente en este repositorio**, por lo que debe copiarse desde la carpeta donde se entrenó el modelo. Sin esos pesos, la interfaz no puede realizar predicciones.

## Entrenar el modelo

La estructura esperada del dataset es:

```text
DetectorGrietas/
├── dataset/
│   ├── Negative/
│   └── Positive/
├── DetectorGrietas.py
├── PruebaDetector.py
├── EvaluarModelo.py
├── EvaluarTodo.py
├── ComplejidadModelo.py
├── LocalizarGrieta.py
├── MedirGrieta.py
├── RiesgoGrieta.py
├── requirements.txt
└── README.md
```

- `Negative`: imágenes de superficies sin grietas.
- `Positive`: imágenes de superficies con grietas.

Coloca las imágenes en sus carpetas correspondientes y ejecuta:

```powershell
python DetectorGrietas.py
```

El script está configurado para separar los datos en entrenamiento (70 %), validación (15 %) y prueba (15 %), con semilla 123, y guarda `modelo_detector_grietas.pth` al terminar. El entrenamiento requiere el dataset completo y puede tardar según el equipo.

## Evaluar el modelo

Con el dataset y los pesos disponibles, ejecuta:

```powershell
python EvaluarModelo.py
python ComplejidadModelo.py
```

La primera herramienta calcula Accuracy, Precision, Recall, F1-score y la matriz de confusión. La segunda estima la cantidad de parámetros, el tamaño del archivo de pesos y el tiempo de inferencia en el equipo donde se ejecuta.

## Limitaciones técnicas

- El clasificador distingue dos clases aprendidas: `Negative` y `Positive`.
- La confianza del modelo no equivale a una probabilidad certificada de daño estructural.
- La localización por OpenCV es una aproximación visual basada en bordes y contornos; puede resaltar textura, juntas, sombras u otros elementos que no sean grietas.
- Las medidas se expresan en píxeles, no en milímetros, salvo que se realice una calibración física específica.
- Un resultado “Negative” no garantiza que la estructura sea segura.

## Tecnologías

Python · PyTorch · torchvision · MobileNetV2 · OpenCV · NumPy · Pillow · scikit-learn · Tkinter
