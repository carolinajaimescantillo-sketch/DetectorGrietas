import cv2
import numpy as np


ROJO_GRIETA = (255, 0, 0)


def calcular_inclinacion(imagen):
    """
    Calcula una inclinación aproximada usando HoughLinesP.
    Se corrige el error de desempaquetado de numpy.int32.
    """

    if imagen is None:
        return 0.0

    if hasattr(imagen, "convert"):
        imagen_cv = np.array(imagen.convert("RGB"))
    else:
        imagen_cv = np.asarray(imagen)

    if imagen_cv.ndim != 3:
        return 0.0

    if imagen_cv.shape[2] == 4:
        imagen_cv = cv2.cvtColor(
            imagen_cv,
            cv2.COLOR_RGBA2RGB
        )

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

    # ESTA ES LA CORRECCIÓN IMPORTANTE
    lineas = np.asarray(lineas).reshape(-1, 4)

    angulos = []

    for x1, y1, x2, y2 in lineas:

        angulo = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1
            )
        )

        if -45 <= angulo <= 45:
            angulos.append(
                abs(float(angulo))
            )

    if len(angulos) == 0:
        return 0.0

    return float(
        np.median(angulos)
    )


def localizar_grieta(imagen):
    """
    Localiza visualmente la grieta utilizando OpenCV.

    Devuelve:

        resultado:
            imagen RGB con la grieta marcada en rojo.

        mascara:
            máscara binaria de la zona detectada.

        evidencia:
            True si se encontró una zona candidata.
    """

    if imagen is None:
        return None, None, False

    if hasattr(imagen, "convert"):
        imagen_cv = np.array(
            imagen.convert("RGB")
        )
    else:
        imagen_cv = np.asarray(imagen).copy()

    if imagen_cv.ndim != 3:
        raise ValueError(
            "La imagen debe tener formato RGB."
        )

    if imagen_cv.shape[2] == 4:
        imagen_cv = cv2.cvtColor(
            imagen_cv,
            cv2.COLOR_RGBA2RGB
        )

    gris = cv2.cvtColor(
        imagen_cv,
        cv2.COLOR_RGB2GRAY
    )

    # Suavizado
    gris_suave = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    # Bordes
    bordes = cv2.Canny(
        gris_suave,
        40,
        120
    )

    # Unir segmentos
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

    # Buscar contornos
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
            (w * h)
            /
            float(ancho * alto)
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
            +
            relacion * 50
            +
            min(
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

    # Si no hay candidatos
    if not candidatos:

        mascara_vacia = np.zeros(
            gris.shape,
            dtype=np.uint8
        )

        return (
            imagen_cv.copy(),
            mascara_vacia,
            False
        )

    # Ordenar candidatos
    candidatos.sort(
        key=lambda elemento: elemento[0],
        reverse=True
    )

    # Tomar los mejores
    candidatos = candidatos[:8]

    # Crear máscara
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
            2
        )

    # Cerrar pequeños espacios
    kernel_pequeno = np.ones(
        (2, 2),
        np.uint8
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel_pequeno
    )

    # Hacer la máscara ligeramente más visible
    kernel_dilatacion = np.ones(
        (2, 2),
        np.uint8
    )

    mascara = cv2.dilate(
        mascara,
        kernel_dilatacion,
        iterations=1
    )

    # Imagen marcada
    resultado = imagen_cv.copy()

    resultado[mascara > 0] = ROJO_GRIETA

    return (
        resultado,
        mascara,
        True
    )


def obtener_ubicacion(mascara):

    if mascara is None:
        return "No determinada"

    if mascara.size == 0:
        return "No determinada"

    ys, xs = np.where(
        mascara > 0
    )

    if len(xs) == 0:
        return "No determinada"

    alto, ancho = mascara.shape

    centro_x = float(
        np.mean(xs)
    )

    centro_y = float(
        np.mean(ys)
    )

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


def porcentaje_mascara(mascara):

    if mascara is None:
        return 0.0

    total = mascara.shape[0] * mascara.shape[1]

    if total <= 0:
        return 0.0

    pixeles = np.count_nonzero(
        mascara > 0
    )

    return (
        pixeles
        /
        float(total)
    ) * 100.0


def medir_mascara(mascara):

    """
    Calcula medidas aproximadas de la zona marcada.
    """

    if mascara is None:
        return {
            "longitud_px": 0.0,
            "ancho_px": 0.0,
            "area_px": 0.0
        }

    ys, xs = np.where(
        mascara > 0
    )

    if len(xs) == 0:
        return {
            "longitud_px": 0.0,
            "ancho_px": 0.0,
            "area_px": 0.0
        }

    longitud = float(
        max(
            xs.max() - xs.min(),
            ys.max() - ys.min()
        )
    )

    ancho = float(
        min(
            xs.max() - xs.min(),
            ys.max() - ys.min()
        )
    )

    area = float(
        np.count_nonzero(
            mascara > 0
        )
    )

    return {
        "longitud_px": longitud,
        "ancho_px": ancho,
        "area_px": area
    }


def evaluar_riesgo(
    crack_score,
    inclinacion
):

    """
    Mantiene la lógica del proyecto:

    < 0.02  -> Sin grieta evidente -> Bajo
    < 0.28  -> Leve               -> Moderado
    < 0.48  -> Advertencia        -> Moderado
    < 0.70  -> Media              -> Alto
    >= 0.70 -> Severa/grave       -> Crítico
    """

    if crack_score < 0.02:

        return (
            "SIN GRIETA EVIDENTE",
            "Bajo",
            "#B7E4C7"
        )

    elif crack_score < 0.28:

        return (
            "LEVE",
            "Moderado",
            "#FFE69C"
        )

    elif crack_score < 0.48:

        return (
            "ADVERTENCIA",
            "Moderado",
            "#FFE69C"
        )

    elif crack_score < 0.70:

        return (
            "MEDIA",
            "Alto",
            "#FFC078"
        )

    else:

        return (
            "SEVERA / GRAVE",
            "Crítico",
            "#FF9B9B"
        )