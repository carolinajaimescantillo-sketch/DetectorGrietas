import cv2
import numpy as np
from tkinter import filedialog


def medir_grieta(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        print("No se pudo abrir la imagen.")
        return

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    bordes = cv2.Canny(
        gris,
        40,
        120
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    bordes = cv2.morphologyEx(
        bordes,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contornos:

        print("No se encontró una posible grieta.")
        return

    mejor_contorno = None
    mejor_puntaje = 0

    for contorno in contornos:

        x, y, w, h = cv2.boundingRect(
            contorno
        )

        longitud = max(w, h)

        relacion = (
            longitud /
            max(min(w, h), 1)
        )

        puntaje = longitud * relacion

        if puntaje > mejor_puntaje:

            mejor_puntaje = puntaje
            mejor_contorno = contorno

    if mejor_contorno is None:

        print("No se pudo localizar la grieta.")
        return

    longitud_pixeles = cv2.arcLength(
        mejor_contorno,
        False
    )

    x, y, w, h = cv2.boundingRect(
        mejor_contorno
    )

    centro_x = x + w // 2
    centro_y = y + h // 2

    alto, ancho = gris.shape

    if centro_x < ancho / 3:
        horizontal = "Izquierda"

    elif centro_x < 2 * ancho / 3:
        horizontal = "Centro"

    else:
        horizontal = "Derecha"

    if centro_y < alto / 3:
        vertical = "Superior"

    elif centro_y < 2 * alto / 3:
        vertical = "Central"

    else:
        vertical = "Inferior"

    ubicacion = (
        f"{vertical} - {horizontal}"
    )

    resultado = imagen.copy()

    cv2.drawContours(
        resultado,
        [mejor_contorno],
        -1,
        (0, 0, 255),
        1
    )

    print()
    print("=" * 50)
    print("MEDICIÓN DE GRIETA")
    print("=" * 50)

    print(
        f"Longitud aproximada: "
        f"{longitud_pixeles:.2f} píxeles"
    )

    print(
        f"Ancho de zona detectada: "
        f"{w} píxeles"
    )

    print(
        f"Alto de zona detectada: "
        f"{h} píxeles"
    )

    print(
        f"Ubicación: {ubicacion}"
    )

    print(
        f"Centro: ({centro_x}, {centro_y})"
    )

    print("=" * 50)

    cv2.imshow(
        "Grieta medida",
        resultado
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


ruta = filedialog.askopenfilename(
    title="Selecciona una imagen con grieta",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png"),
        ("Todos los archivos", "*.*")
    ]
)

if ruta:

    medir_grieta(ruta)