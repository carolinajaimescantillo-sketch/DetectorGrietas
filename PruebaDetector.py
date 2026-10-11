import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import torch
from torchvision import models, transforms
import torch.nn as nn

import cv2
import numpy as np
import time
import os


IMG_SIZE = (224, 224)

CARPETA_PROYECTO = os.path.dirname(
    os.path.abspath(__file__)
)

MODELO = os.path.join(
    CARPETA_PROYECTO,
    "modelo_detector_grietas.pth"
)

CLASES = [
    "Negative",
    "Positive"
]


FONDO = "#F9F3FA"
BLANCO = "#FFFFFF"

ROSA_CLARO = "#FCE4F1"
ROSA_BORDE = "#F4B6D7"

MORADO = "#7B4BA3"
MORADO_OSCURO = "#5B2C83"

MORADO_DATOS = "#8E5BB7"
GRIS = "#777777"

VERDE_APAGADO = "#D5E8D4"
VERDE_ENCENDIDO = "#4CAF50"

AMARILLO_APAGADO = "#FFF0B8"
AMARILLO_ENCENDIDO = "#FFD43B"

NARANJA_APAGADO = "#FFE0B2"
NARANJA_ENCENDIDO = "#F7C98B"

ROJO_APAGADO = "#FFD1D1"
ROJO_ENCENDIDO = "#F4A3A3"

ROJO_GRIETA = (255, 0, 60)


model = models.mobilenet_v2(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    2
)


try:

    model.load_state_dict(
        torch.load(
            MODELO,
            map_location="cpu"
        )
    )

    model.eval()

except Exception as error:

    ventana_error = tk.Tk()
    ventana_error.withdraw()

    messagebox.showerror(
        "Error del modelo",
        "No se pudo cargar el modelo.\n\n"
        f"Ruta buscada:\n{MODELO}\n\n"
        f"Detalle:\n{error}"
    )

    ventana_error.destroy()

    raise SystemExit


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])


def calcular_inclinacion(imagen):

    imagen_cv = np.array(imagen)

    gris = cv2.cvtColor(
        imagen_cv,
        cv2.COLOR_RGB2GRAY
    )

    bordes = cv2.Canny(
        gris,
        50,
        150
    )

    lineas = cv2.HoughLinesP(
        bordes,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    if lineas is None:

        return 0.0

    angulos = []

    for linea in lineas:

        x1, y1, x2, y2 = linea[0]

        angulo = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        if -45 <= angulo <= 45:

            angulos.append(
                abs(angulo)
            )

    if len(angulos) == 0:

        return 0.0

    return float(
        np.median(angulos)
    )


def evaluar_riesgo(
    crack_score,
    inclinacion,
    clase
):

    if clase == "Negative":

        return {
            "nivel": "SIN GRIETA EVIDENTE",
            "riesgo": "Bajo",
            "color": VERDE_ENCENDIDO
        }

    if crack_score < 0.40 and inclinacion < 10:

        return {
            "nivel": "LEVE",
            "riesgo": "Bajo",
            "color": VERDE_ENCENDIDO
        }

    elif crack_score < 0.60 and inclinacion < 15:

        return {
            "nivel": "ADVERTENCIA",
            "riesgo": "Moderado",
            "color": AMARILLO_ENCENDIDO
        }

    elif crack_score < 0.80 and inclinacion < 25:

        return {
            "nivel": "MEDIA",
            "riesgo": "Alto",
            "color": NARANJA_ENCENDIDO
        }

    else:

        return {
            "nivel": "SEVERA / GRAVE",
            "riesgo": "Crítico",
            "color": ROJO_ENCENDIDO
        }


def localizar_grieta(imagen):

    imagen_cv = np.array(
        imagen
    ).copy()

    gris = cv2.cvtColor(
        imagen_cv,
        cv2.COLOR_RGB2GRAY
    )

    gris_suave = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        gris_suave,
        40,
        120
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    bordes_unidos = cv2.morphologyEx(
        bordes,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        bordes_unidos,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    candidatos = []

    alto, ancho = gris.shape

    for contorno in contornos:

        area = cv2.contourArea(
            contorno
        )

        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(
            contorno
        )

        if w < 8 or h < 8:
            continue

        porcentaje_imagen = (
            (w * h) /
            (ancho * alto)
        )

        if porcentaje_imagen > 0.70:
            continue

        relacion = max(
            w / max(h, 1),
            h / max(w, 1)
        )

        if relacion < 1.4:
            continue

        perimetro = cv2.arcLength(
            contorno,
            False
        )

        if perimetro <= 0:
            continue

        longitud = max(
            w,
            h
        )

        puntaje = (
            longitud * 2
            + relacion * 50
            + min(
                perimetro,
                1000
            ) * 0.3
        )

        candidatos.append(
            (
                puntaje,
                contorno
            )
        )

    if not candidatos:

        mascara_vacia = np.zeros(
            gris.shape,
            dtype=np.uint8
        )

        return (
            imagen_cv,
            mascara_vacia
        )

    candidatos.sort(
        key=lambda elemento: elemento[0],
        reverse=True
    )

    candidatos = candidatos[:8]

    mascara = np.zeros(
        gris.shape,
        dtype=np.uint8
    )

    for _, contorno in candidatos:

        cv2.drawContours(
            mascara,
            [contorno],
            -1,
            255,
            3
        )

    kernel_pequeno = np.ones(
        (3, 3),
        np.uint8
    )

    mascara = cv2.dilate(
        mascara,
        kernel_pequeno,
        iterations=1
    )

    resultado = imagen_cv.copy()

    resultado[
        mascara > 0
    ] = ROJO_GRIETA

    return (
        resultado,
        mascara
    )


def obtener_ubicacion(mascara):

    if mascara is None:

        return "No determinada"

    ys, xs = np.where(
        mascara > 0
    )

    if len(xs) == 0:

        return "No determinada"

    alto, ancho = mascara.shape

    centro_x = np.mean(xs)
    centro_y = np.mean(ys)

    if centro_x < ancho / 3:

        horizontal = "Izquierda"

    elif centro_x < (2 * ancho / 3):

        horizontal = "Centro"

    else:

        horizontal = "Derecha"

    if centro_y < alto / 3:

        vertical = "Superior"

    elif centro_y < (2 * alto / 3):

        vertical = "Central"

    else:

        vertical = "Inferior"

    return (
        f"{vertical} - {horizontal}"
    )


ventana = tk.Tk()

ventana.title(
    "Detector de Grietas"
)

ventana.geometry(
    "1050x850"
)

ventana.configure(
    bg=FONDO
)

ventana.resizable(
    True,
    True
)


contenedor_principal = tk.Frame(
    ventana,
    bg=FONDO
)

contenedor_principal.pack(
    fill="both",
    expand=True
)


canvas = tk.Canvas(
    contenedor_principal,
    bg=FONDO,
    highlightthickness=0
)

barra_desplazamiento = tk.Scrollbar(
    contenedor_principal,
    orient="vertical",
    command=canvas.yview
)

contenido = tk.Frame(
    canvas,
    bg=FONDO
)


ventana_canvas = canvas.create_window(
    (0, 0),
    window=contenido,
    anchor="nw"
)


def actualizar_scroll(event):

    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


contenido.bind(
    "<Configure>",
    actualizar_scroll
)


def ajustar_ancho(event):

    canvas.itemconfig(
        ventana_canvas,
        width=event.width
    )


canvas.bind(
    "<Configure>",
    ajustar_ancho
)


canvas.configure(
    yscrollcommand=barra_desplazamiento.set
)


canvas.pack(
    side="left",
    fill="both",
    expand=True
)

barra_desplazamiento.pack(
    side="right",
    fill="y"
)


def desplazar_rueda(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    desplazar_rueda
)


encabezado = tk.Frame(
    contenido,
    bg=FONDO
)

encabezado.pack(
    pady=(18, 5)
)


titulo = tk.Label(
    encabezado,
    text="DETECTOR DE GRIETAS",
    font=("Georgia", 30, "bold"),
    bg=FONDO,
    fg=MORADO_OSCURO
)

titulo.pack()


subtitulo = tk.Label(
    encabezado,
    text="Análisis inteligente de superficies",
    font=("Georgia", 15, "italic"),
    bg=FONDO,
    fg=MORADO
)

subtitulo.pack(
    pady=(3, 0)
)


linea = tk.Frame(
    contenido,
    bg=ROSA_BORDE,
    height=2,
    width=1000
)

linea.pack(
    pady=(5, 10)
)


marco_imagen_exterior = tk.Frame(
    contenido,
    bg=ROSA_BORDE,
    padx=3,
    pady=3
)

marco_imagen_exterior.pack(
    padx=30,
    pady=(0, 10)
)


marco_imagen = tk.Frame(
    marco_imagen_exterior,
    bg=BLANCO,
    width=1000,
    height=390
)

marco_imagen.pack()

marco_imagen.pack_propagate(
    False
)


titulo_imagen = tk.Label(
    marco_imagen,
    text="IMÁGENES PARA INSPECCIÓN",
    font=("Georgia", 14, "bold"),
    bg=BLANCO,
    fg=MORADO
)

titulo_imagen.pack(
    pady=(10, 5)
)


contenedor_original = tk.Frame(
    marco_imagen,
    bg=BLANCO,
    width=470,
    height=330,
    highlightbackground="#CCCCCC",
    highlightthickness=1
)

contenedor_original.pack(
    side="left",
    padx=(20, 10)
)

contenedor_original.pack_propagate(
    False
)


contenedor_marcada = tk.Frame(
    marco_imagen,
    bg=BLANCO,
    width=470,
    height=330,
    highlightbackground="#CCCCCC",
    highlightthickness=1
)

contenedor_marcada.pack(
    side="left",
    padx=(10, 20)
)

contenedor_marcada.pack_propagate(
    False
)


titulo_original = tk.Label(
    contenedor_original,
    text="ORIGINAL",
    font=("Georgia", 12, "bold"),
    bg=BLANCO,
    fg=MORADO_OSCURO
)

titulo_original.pack(
    pady=(8, 5)
)


titulo_marcada = tk.Label(
    contenedor_marcada,
    text="GRIETA MARCADA",
    font=("Georgia", 12, "bold"),
    bg=BLANCO,
    fg=MORADO_OSCURO
)

titulo_marcada.pack(
    pady=(8, 5)
)


imagen_original_label = tk.Label(
    contenedor_original,
    text="Seleccione una imagen\npara iniciar el análisis",
    font=("Georgia", 12),
    bg=BLANCO,
    fg="#999999"
)

imagen_original_label.pack(
    expand=True
)


imagen_marcada_label = tk.Label(
    contenedor_marcada,
    text="Aparecerá después del análisis",
    font=("Georgia", 12),
    bg=BLANCO,
    fg="#999999"
)

imagen_marcada_label.pack(
    expand=True
)


marco_resultado = tk.Frame(
    contenido,
    bg=ROSA_CLARO,
    highlightbackground=ROSA_BORDE,
    highlightthickness=2
)

marco_resultado.pack(
    padx=100,
    fill="x",
    pady=(0, 8)
)


titulo_resultado = tk.Label(
    marco_resultado,
    text="RESULTADO DEL ANÁLISIS",
    font=("Georgia", 13, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO
)

titulo_resultado.pack(
    pady=(7, 5)
)


resultado_contenedor = tk.Frame(
    marco_resultado,
    bg=ROSA_CLARO
)

resultado_contenedor.pack(
    padx=25,
    pady=(0, 7)
)


datos_resultado = {}

filas_resultado = [
    "Estado",
    "Clase",
    "Confianza",
    "Nivel",
    "Riesgo",
    "Probabilidad de grieta",
    "Inclinación",
    "Ubicación",
    "Tiempo de inferencia"
]


for fila, nombre in enumerate(filas_resultado):

    etiqueta_nombre = tk.Label(
        resultado_contenedor,
        text=f"{nombre}:",
        font=("Georgia", 9, "bold"),
        bg=ROSA_CLARO,
        fg=MORADO,
        anchor="e",
        width=25
    )

    etiqueta_nombre.grid(
        row=fila,
        column=0,
        padx=(5, 8),
        pady=2,
        sticky="e"
    )

    variable = tk.StringVar(
        value="-"
    )

    etiqueta_valor = tk.Label(
        resultado_contenedor,
        textvariable=variable,
        font=("Georgia", 9, "bold"),
        bg=ROSA_CLARO,
        fg=MORADO_DATOS,
        anchor="w",
        width=30
    )

    etiqueta_valor.grid(
        row=fila,
        column=1,
        padx=(8, 5),
        pady=2,
        sticky="w"
    )

    datos_resultado[nombre] = (
        variable,
        etiqueta_valor
    )


marco_semaforo = tk.Frame(
    contenido,
    bg=BLANCO,
    highlightbackground=ROSA_BORDE,
    highlightthickness=2,
    width=600,
    height=155
)

marco_semaforo.pack(
    pady=(0, 8)
)

marco_semaforo.pack_propagate(
    False
)


titulo_semaforo = tk.Label(
    marco_semaforo,
    text="NIVEL DE RIESGO",
    font=("Georgia", 14, "bold"),
    bg=BLANCO,
    fg=MORADO_OSCURO
)

titulo_semaforo.pack(
    pady=(8, 3)
)


semaforo_contenedor = tk.Frame(
    marco_semaforo,
    bg=BLANCO
)

semaforo_contenedor.pack()


canvas_semaforo = tk.Canvas(
    semaforo_contenedor,
    width=500,
    height=75,
    bg=BLANCO,
    highlightthickness=0
)

canvas_semaforo.pack()


luz_verde = canvas_semaforo.create_oval(
    35,
    5,
    95,
    65,
    fill=VERDE_APAGADO,
    outline=ROSA_BORDE,
    width=2
)


luz_amarilla = canvas_semaforo.create_oval(
    155,
    5,
    215,
    65,
    fill=AMARILLO_APAGADO,
    outline=ROSA_BORDE,
    width=2
)


luz_naranja = canvas_semaforo.create_oval(
    275,
    5,
    335,
    65,
    fill=NARANJA_APAGADO,
    outline=ROSA_BORDE,
    width=2
)


luz_roja = canvas_semaforo.create_oval(
    395,
    5,
    455,
    65,
    fill=ROJO_APAGADO,
    outline=ROSA_BORDE,
    width=2
)


etiquetas_semaforo = tk.Frame(
    semaforo_contenedor,
    bg=BLANCO
)

etiquetas_semaforo.pack()


tk.Label(
    etiquetas_semaforo,
    text="BAJO",
    font=("Georgia", 9, "bold"),
    bg=BLANCO,
    fg="#4CAF50",
    width=12
).pack(
    side="left"
)


tk.Label(
    etiquetas_semaforo,
    text="MODERADO",
    font=("Georgia", 9, "bold"),
    bg=BLANCO,
    fg="#D69E00",
    width=12
).pack(
    side="left"
)


tk.Label(
    etiquetas_semaforo,
    text="ALTO",
    font=("Georgia", 9, "bold"),
    bg=BLANCO,
    fg="#E67E00",
    width=12
).pack(
    side="left"
)


tk.Label(
    etiquetas_semaforo,
    text="CRÍTICO",
    font=("Georgia", 9, "bold"),
    bg=BLANCO,
    fg="#E53935",
    width=12
).pack(
    side="left"
)


estado_semaforo = tk.Label(
    marco_semaforo,
    text="Esperando análisis",
    font=("Georgia", 10, "bold"),
    bg=BLANCO,
    fg=GRIS
)

estado_semaforo.pack(
    pady=(2, 0)
)


def actualizar_semaforo(riesgo):

    canvas_semaforo.itemconfig(
        luz_verde,
        fill=VERDE_APAGADO
    )

    canvas_semaforo.itemconfig(
        luz_amarilla,
        fill=AMARILLO_APAGADO
    )

    canvas_semaforo.itemconfig(
        luz_naranja,
        fill=NARANJA_APAGADO
    )

    canvas_semaforo.itemconfig(
        luz_roja,
        fill=ROJO_APAGADO
    )

    if riesgo == "Bajo":

        canvas_semaforo.itemconfig(
            luz_verde,
            fill=VERDE_ENCENDIDO
        )

        estado_semaforo.config(
            text="RIESGO BAJO",
            fg=VERDE_ENCENDIDO
        )

    elif riesgo == "Moderado":

        canvas_semaforo.itemconfig(
            luz_amarilla,
            fill=AMARILLO_ENCENDIDO
        )

        estado_semaforo.config(
            text="RIESGO MODERADO",
            fg="#D69E00"
        )

    elif riesgo == "Alto":

        canvas_semaforo.itemconfig(
            luz_naranja,
            fill=NARANJA_ENCENDIDO
        )

        estado_semaforo.config(
            text="RIESGO ALTO",
            fg="#E67E00"
        )

    elif riesgo == "Crítico":

        canvas_semaforo.itemconfig(
            luz_roja,
            fill=ROJO_ENCENDIDO
        )

        estado_semaforo.config(
            text="RIESGO CRÍTICO",
            fg="#D94A4A"
        )


marco_evaluacion = tk.Frame(
    contenido,
    bg=ROSA_CLARO,
    highlightbackground=ROSA_BORDE,
    highlightthickness=2
)

marco_evaluacion.pack(
    padx=100,
    fill="x",
    pady=(0, 8)
)


titulo_evaluacion = tk.Label(
    marco_evaluacion,
    text="RENDIMIENTO DEL MODELO",
    font=("Georgia", 13, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO
)

titulo_evaluacion.pack(
    pady=(7, 4)
)


datos_evaluacion = tk.Label(
    marco_evaluacion,
    text=(
        "Accuracy: 99,32 %    •    "
        "Precision: 99,66 %    •    "
        "Recall: 98,94 %\n"
        "F1-score: 99,30 %    •    "
        "Imágenes evaluadas: 6.000\n"
        "Correctas: 5.959    •    "
        "Incorrectas: 41"
    ),
    font=("Georgia", 9),
    bg=ROSA_CLARO,
    fg=MORADO
)

datos_evaluacion.pack(
    pady=(0, 7)
)


marco_confusion = tk.Frame(
    contenido,
    bg=BLANCO,
    highlightbackground=ROSA_BORDE,
    highlightthickness=2
)

marco_confusion.pack(
    padx=100,
    fill="x",
    pady=(0, 8)
)


titulo_confusion = tk.Label(
    marco_confusion,
    text="MATRIZ DE CONFUSIÓN",
    font=("Georgia", 13, "bold"),
    bg=BLANCO,
    fg=MORADO_OSCURO
)

titulo_confusion.pack(
    pady=(7, 5)
)


tabla_confusion = tk.Frame(
    marco_confusion,
    bg=BLANCO
)

tabla_confusion.pack(
    pady=(0, 8)
)


tk.Label(
    tabla_confusion,
    text="",
    font=("Georgia", 9, "bold"),
    bg=BLANCO,
    width=15
).grid(
    row=0,
    column=0,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="Predicted Negative",
    font=("Georgia", 9, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=0,
    column=1,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="Predicted Positive",
    font=("Georgia", 9, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=0,
    column=2,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="Actual Negative",
    font=("Georgia", 9, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=15
).grid(
    row=1,
    column=0,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="3053",
    font=("Georgia", 11, "bold"),
    bg=VERDE_APAGADO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=1,
    column=1,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="10",
    font=("Georgia", 11, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=1,
    column=2,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="Actual Positive",
    font=("Georgia", 9, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=15
).grid(
    row=2,
    column=0,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="31",
    font=("Georgia", 11, "bold"),
    bg=ROSA_CLARO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=2,
    column=1,
    padx=3,
    pady=3
)


tk.Label(
    tabla_confusion,
    text="2906",
    font=("Georgia", 11, "bold"),
    bg=VERDE_APAGADO,
    fg=MORADO_OSCURO,
    width=20
).grid(
    row=2,
    column=2,
    padx=3,
    pady=3
)


boton = tk.Button(
    contenido,
    text="SELECCIONAR IMAGEN",
    command=lambda: seleccionar_imagen(),
    font=("Georgia", 13, "bold"),
    bg=MORADO,
    fg=BLANCO,
    activebackground=MORADO_OSCURO,
    activeforeground=BLANCO,
    relief="flat",
    cursor="hand2",
    padx=35,
    pady=10
)

boton.pack(
    pady=(3, 8)
)


aviso = tk.Label(
    contenido,
    text=(
        "Aviso: este resultado es una evaluación visual preliminar "
        "y no sustituye una inspección estructural profesional."
    ),
    font=("Georgia", 8, "italic"),
    bg=FONDO,
    fg=GRIS,
    wraplength=850,
    justify="center"
)

aviso.pack(
    pady=(2, 3)
)


info = tk.Label(
    contenido,
    text=(
        "Clasificación automática  •  "
        "MobileNetV2  •  "
        "OpenCV  •  "
        "Inteligencia Artificial"
    ),
    font=("Georgia", 8),
    bg=FONDO,
    fg=GRIS
)

info.pack(
    pady=(1, 2)
)


firma = tk.Label(
    contenido,
    text="Detector de Grietas",
    font=("Georgia", 9, "italic"),
    bg=FONDO,
    fg=MORADO
)

firma.pack(
    pady=(0, 10)
)


def actualizar_resultado(
    estado,
    clase,
    confianza,
    nivel,
    riesgo,
    probabilidad,
    inclinacion,
    ubicacion,
    tiempo,
    color
):

    datos = {
        "Estado": estado,
        "Clase": clase,
        "Confianza": f"{confianza:.2f} %",
        "Nivel": nivel,
        "Riesgo": riesgo,
        "Probabilidad de grieta": f"{probabilidad:.2f} %",
        "Inclinación": f"{inclinacion:.2f}°",
        "Ubicación": ubicacion,
        "Tiempo de inferencia": f"{tiempo:.2f} ms"
    }

    marco_resultado.config(
        bg=color
    )

    titulo_resultado.config(
        bg=color
    )

    resultado_contenedor.config(
        bg=color
    )

    for nombre, valor in datos.items():

        variable, etiqueta = datos_resultado[nombre]

        variable.set(
            valor
        )

        etiqueta.config(
            bg=color
        )

    for nombre in datos_resultado:

        variable, etiqueta = datos_resultado[nombre]

        etiqueta.config(
            bg=color
        )


def seleccionar_imagen():

    ruta = filedialog.askopenfilename(

        title="Selecciona una imagen",

        filetypes=[
            (
                "Imágenes",
                "*.jpg *.jpeg *.png *.bmp"
            ),
            (
                "Todos los archivos",
                "*.*"
            )
        ]
    )

    if not ruta:

        return

    try:

        imagen_original = Image.open(
            ruta
        ).convert("RGB")


        inclinacion = calcular_inclinacion(
            imagen_original
        )


        imagen_modelo = transform(
            imagen_original
        )

        imagen_modelo = imagen_modelo.unsqueeze(
            0
        )


        inicio = time.perf_counter()

        with torch.no_grad():

            salida = model(
                imagen_modelo
            )

            probabilidades = torch.softmax(
                salida,
                dim=1
            )

            prediccion = torch.argmax(
                probabilidades,
                dim=1
            )

        fin = time.perf_counter()


        tiempo_inferencia = (
            fin - inicio
        ) * 1000


        indice_clase = prediccion.item()

        clase = CLASES[
            indice_clase
        ]


        confianza = (
            probabilidades[0][indice_clase].item()
            * 100
        )


        crack_score = (
            probabilidades[0][1].item()
        )


        resultado_riesgo = evaluar_riesgo(
            crack_score,
            inclinacion,
            clase
        )


        nivel_texto = resultado_riesgo[
            "nivel"
        ]

        nivel_riesgo = resultado_riesgo[
            "riesgo"
        ]

        color_riesgo = resultado_riesgo[
            "color"
        ]


        imagen_marcada = np.array(
            imagen_original
        ).copy()


        mascara = np.zeros(
            (
                imagen_marcada.shape[0],
                imagen_marcada.shape[1]
            ),
            dtype=np.uint8
        )


        ubicacion = "No aplica"


        if clase == "Positive":

            resultado_localizacion = localizar_grieta(
                imagen_original
            )

            imagen_marcada = (
                resultado_localizacion[0]
            )

            mascara = (
                resultado_localizacion[1]
            )

            ubicacion = obtener_ubicacion(
                mascara
            )


        imagen_original_mostrar = (
            imagen_original.copy()
        )

        imagen_original_mostrar.thumbnail(
            (320, 200)
        )


        imagen_original_tk = ImageTk.PhotoImage(
            imagen_original_mostrar
        )


        imagen_original_label.config(
            image=imagen_original_tk,
            text=""
        )

        imagen_original_label.image = (
            imagen_original_tk
        )


        if clase == "Positive":

            if isinstance(
                imagen_marcada,
                np.ndarray
            ):

                imagen_marcada_pil = (
                    Image.fromarray(
                        imagen_marcada
                    )
                )

            else:

                imagen_marcada_pil = (
                    imagen_marcada.copy()
                )


            imagen_marcada_pil.thumbnail(
                (320, 200)
            )


            imagen_marcada_tk = ImageTk.PhotoImage(
                imagen_marcada_pil
            )


            imagen_marcada_label.config(
                image=imagen_marcada_tk,
                text=""
            )


            imagen_marcada_label.image = (
                imagen_marcada_tk
            )

        else:

            imagen_marcada_label.config(
                image="",
                text="NO SE DETECTÓ GRIETA",
                font=("Georgia", 14, "bold"),
                fg=VERDE_ENCENDIDO,
                bg=BLANCO
            )

            imagen_marcada_label.image = None


        if clase == "Positive":

            estado = "POSIBLE GRIETA DETECTADA"

        else:

            estado = "NO SE DETECTÓ UNA GRIETA"


        actualizar_resultado(
            estado,
            clase,
            confianza,
            nivel_texto,
            nivel_riesgo,
            crack_score * 100,
            inclinacion,
            ubicacion,
            tiempo_inferencia,
            color_riesgo
        )


        actualizar_semaforo(
            nivel_riesgo
        )


        print()
        print("=" * 55)
        print("ANÁLISIS DE IMAGEN")
        print("=" * 55)

        print(
            f"Clase: {clase}"
        )

        print(
            f"Confianza: {confianza:.2f}%"
        )

        print(
            f"Probabilidad de grieta: "
            f"{crack_score * 100:.2f}%"
        )

        print(
            f"Inclinación: "
            f"{inclinacion:.2f}°"
        )

        print(
            f"Nivel: "
            f"{nivel_texto}"
        )

        print(
            f"Riesgo: "
            f"{nivel_riesgo}"
        )

        print(
            f"Ubicación: "
            f"{ubicacion}"
        )

        print(
            f"Tiempo de inferencia: "
            f"{tiempo_inferencia:.2f} ms"
        )

        print("=" * 55)


    except Exception as error:

        messagebox.showerror(
            "Error",
            "No fue posible analizar la imagen.\n\n"
            f"{error}"
        )

        print()
        print("ERROR DURANTE EL ANÁLISIS:")
        print(error)


ventana.mainloop()