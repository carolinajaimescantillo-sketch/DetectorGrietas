import tkinter as tk


FONDO = "#F9F3FA"
ROSA_BORDE = "#F4B6D7"
MORADO_OSCURO = "#5B2C83"
BLANCO = "#FFFFFF"


def mostrar_semaforo(riesgo):

    ventana = tk.Tk()

    ventana.title("Evaluación de riesgo")
    ventana.geometry("320x500")
    ventana.configure(bg=FONDO)
    ventana.resizable(False, False)


    titulo = tk.Label(
        ventana,
        text="NIVEL DE RIESGO",
        font=("Georgia", 18, "bold"),
        bg=FONDO,
        fg=MORADO_OSCURO
    )

    titulo.pack(pady=(25, 15))

    marco = tk.Frame(
        ventana,
        bg=BLANCO,
        highlightbackground=ROSA_BORDE,
        highlightthickness=2
    )

    marco.pack(
        padx=30,
        pady=5
    )

    canvas = tk.Canvas(
        marco,
        width=220,
        height=350,
        bg=BLANCO,
        highlightthickness=0
    )

    canvas.pack(
        padx=20,
        pady=20
    )

    canvas.create_rectangle(
        50,
        20,
        170,
        330,
        fill=MORADO_OSCURO,
        outline=MORADO_OSCURO
    )

    rojo_apagado = "#7A4A55"
    amarillo_apagado = "#8A7650"
    verde_apagado = "#4D765C"


    luz_roja = canvas.create_oval(
        75,
        40,
        145,
        110,
        fill=rojo_apagado,
        outline=ROSA_BORDE,
        width=2
    )

    luz_amarilla = canvas.create_oval(
        75,
        135,
        145,
        205,
        fill=amarillo_apagado,
        outline=ROSA_BORDE,
        width=2
    )

    luz_verde = canvas.create_oval(
        75,
        230,
        145,
        300,
        fill=verde_apagado,
        outline=ROSA_BORDE,
        width=2
    )

    if riesgo == "Bajo":

        canvas.itemconfig(
            luz_verde,
            fill="#69D38A"
        )

        texto_riesgo = "RIESGO BAJO"

    elif riesgo == "Moderado":

        canvas.itemconfig(
            luz_amarilla,
            fill="#FFD84D"
        )

        texto_riesgo = "RIESGO MODERADO"

    elif riesgo == "Alto":

        canvas.itemconfig(
            luz_roja,
            fill="#FF4D5A"
        )

        texto_riesgo = "RIESGO ALTO"

    elif riesgo == "Crítico":

        canvas.itemconfig(
            luz_roja,
            fill="#FF1F35"
        )

        texto_riesgo = "RIESGO CRÍTICO"

    else:

        texto_riesgo = "RIESGO NO DETERMINADO"

    resultado = tk.Label(
        ventana,
        text=texto_riesgo,
        font=("Georgia", 15, "bold"),
        bg=FONDO,
        fg=MORADO_OSCURO
    )

    resultado.pack(
        pady=(10, 5)
    )

    ventana.mainloop()

mostrar_semaforo("Bajo")