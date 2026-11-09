# DATOS DE ENTRENAMIENTO
# Cada tupla es (x1, x2, t) donde t es la salida deseada (target).
DATOS_AND = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]



# FUNCION DE ACTIVACION (escalon)
# Recibe z (la suma ponderada) y devuelve 1 o 0.
# Regla: si z es mayor o igual a 0 -> 1, en otro caso -> 0.
def escalon(z):
    return 1 if z >= 0 else 0


class Perceptron:
    def __init__(self, w1, w2, b, eta):
        self.w1 = w1          # peso de la entrada x1
        self.w2 = w2          # peso de la entrada x2
        self.b = b            # bias (sesgo)
        self.eta = eta        # tasa de aprendizaje

        # Historial detallado: un diccionario por cada muestra de cada epoca.
        self.pasos = []

        # Historial resumido: un diccionario por epoca con los pesos finales
        # y cuantos errores hubo.
        self.epocas = []

    # PREDECIR
    # Calcula la salida del perceptron con los pesos ACTUALES.
    # No modifica los pesos.
    # Devuelve dos cosas: z (suma ponderada) y y (salida 0 o 1).
    def predecir(self, x1, x2):
        # Suma ponderada: z = x1*w1 + x2*w2 + b
        z = x1 * self.w1 + x2 * self.w2 + self.b

        # Redondeo para eliminar el ruido de los decimales
        # (por ejemplo -2.7e-17 en lugar de 0). Asi z = 0 se queda como 0.
        z = round(z, 10)

        return z, escalon(z)

    # ENTRENAR
    # Recorre las 4 muestras una y otra vez (cada vuelta es una epoca)
    # y ajusta los pesos cada vez que se equivoca.
    # Se detiene cuando una epoca completa no tiene errores,
    # o cuando llega al limite de epocas.
    def entrenar(self, max_epocas):
        # Se limpian los historiales por si se entrena mas de una vez
        self.pasos = []
        self.epocas = []

        # Cada vuelta de este for es UNA EPOCA (empieza en 1)
        for epoca in range(1, max_epocas + 1):
            errores = 0   # contador de errores de esta epoca

            # Cada vuelta de este for es UNA MUESTRA de la tabla AND
            for x1, x2, t in DATOS_AND:

                # Se guardan los valores ANTES de modificarlos,
                # porque el procedimiento paso a paso necesita mostrarlos
                w1_ant, w2_ant, b_ant = self.w1, self.w2, self.b

                # 1) Calcular la salida actual: z y y = f(z)
                z, y = self.predecir(x1, x2)

                # 2) Calcular el error: deseado menos obtenido
                #    e = 0 acierto, e = 1 faltaba activar, e = -1 se activo de mas
                e = t - y

                # 3) Regla delta: w_nuevo = w_anterior + eta * e * x
                #    Si e = 0 el peso no cambia.
                #    Se redondea a 10 decimales por el mismo motivo de arriba.
                self.w1 = round(w1_ant + self.eta * e * x1, 10)
                self.w2 = round(w2_ant + self.eta * e * x2, 10)

                #    El bias se actualiza igual, pero como si su entrada fuera
                #    siempre 1, por eso no se multiplica por ninguna x.
                self.b = round(b_ant + self.eta * e, 10)

                # 4) Contar el error si hubo uno
                if e != 0:
                    errores += 1

                # 5) Guardar todo en esta muestra (para mostrarlo luego)
                self.pasos.append({
                    "epoca": epoca,
                    "x1": x1, "x2": x2, "t": t,
                    "w1_ant": w1_ant, "w2_ant": w2_ant, "b_ant": b_ant,
                    "z": z, "y": y, "e": e,
                    "w1_nuevo": self.w1, "w2_nuevo": self.w2, "b_nuevo": self.b,
                })

            # Termino la epoca: se guarda el resumen con los pesos finales
            self.epocas.append({
                "epoca": epoca,
                "w1": self.w1, "w2": self.w2, "b": self.b,
                "errores": errores,
            })

            # Si en toda la epoca no hubo errores, ya aprendio: se detiene
            if errores == 0:
                break



# prueba rapida, este lo ignora el app.py
if __name__ == "__main__":
    p = Perceptron(w1=0, w2=0, b=0, eta=0.1)
    p.entrenar(max_epocas=20)
    for ep in p.epocas:
        print(ep)