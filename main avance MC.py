import hashlib
import json
import time
from pathlib import Path

import numpy as np
from arduino.app_utils import App, Bridge

# Requiere el TFLite y el JSON exportados por el cuaderno de tres clases.
MODO_PRUEBA = False
CAMARA = 0
NOMBRES_LED = {0: "APAGADOS", 1: "VERDE", 2: "NARANJA", 3: "ROJO"}
estado_confirmado = None
ultimo_envio = 0.0
paso_prueba = 0


def enviar_estado(estado, forzar=False):
    global estado_confirmado, ultimo_envio
    if (not forzar and estado == estado_confirmado
            and time.monotonic() - ultimo_envio < 1.0):
        return True
    try:
        respuesta = Bridge.call("establecer_estado", int(estado), timeout=2)
        if type(respuesta) is not int or respuesta != estado:
            raise RuntimeError(f"Respuesta inesperada: {respuesta!r}")
    except Exception as error:
        print(f"ERROR de comunicacion: {error}", flush=True)
        time.sleep(0.5)
        return False
    if estado != estado_confirmado:
        print(f"STM32 confirmo: {NOMBRES_LED[estado]}", flush=True)
    estado_confirmado = estado
    ultimo_envio = time.monotonic()
    return True


def loop_prueba():
    global paso_prueba
    if enviar_estado((1, 2, 3, 0)[paso_prueba], forzar=True):
        paso_prueba = (paso_prueba + 1) % 4
    time.sleep(1)


class FiltroFiguras:
    """Rechazo provisional con positivos; no certifica ausencia de figura."""

    def __init__(self, config):
        self.clases = tuple(config["class_names"])
        self.estados = tuple(config["led_states"])
        if (self.clases != ("grave", "herido", "ileso")
                or self.estados != (3, 2, 1)):
            raise ValueError("Correspondencia de clases/LED incompatible.")
        rechazo = config["rejection"]
        if rechazo.get("method") != "class_cosine_prototypes_v1":
            raise ValueError("Metodo de rechazo incompatible.")
        self.centros = np.asarray(rechazo["prototypes"], dtype=np.float32)
        self.etiquetas = np.asarray(rechazo["prototype_labels"], dtype=np.int32)
        self.limites = np.asarray(rechazo["distance_limits"], dtype=np.float32)
        self.prob_min = float(rechazo["probability_min"])
        self.repeticiones = int(rechazo["consecutive_frames"])
        if (self.centros.ndim != 2 or self.centros.shape[1] < 4
                or len(self.etiquetas) != len(self.centros)
                or set(self.etiquetas.tolist()) != {0, 1, 2}
                or self.limites.shape != (3,)
                or not np.all(np.isfinite(self.centros))
                or not np.all(np.isfinite(self.limites))
                or np.any(self.limites <= 0) or np.any(self.limites > 2)
                or not 0 < self.prob_min <= 1 or self.repeticiones < 1):
            raise ValueError("Configuracion del filtro no valida.")
        normas = np.linalg.norm(self.centros, axis=1, keepdims=True)
        if np.any(normas < 1e-8):
            raise ValueError("Hay un prototipo sin informacion.")
        self.centros = self.centros / normas
        self.reiniciar()

    def reiniciar(self):
        self.candidata = None
        self.contador = 0

    def evaluar(self, probabilidades, caracteristicas):
        p = np.asarray(probabilidades, dtype=np.float32).reshape(-1)
        f = np.asarray(caracteristicas, dtype=np.float32).reshape(-1)
        if (p.shape != (3,) or f.shape != (self.centros.shape[1],)
                or not np.all(np.isfinite(p)) or not np.all(np.isfinite(f))
                or np.any(p < 0) or np.any(p > 1)
                or not np.isclose(p.sum(), 1.0, atol=0.02)):
            self.reiniciar()
            return 0, "SALIDA INVALIDA"
        indice = int(np.argmax(p))
        confianza = float(p[indice])
        norma = float(np.linalg.norm(f))
        if norma < 1e-8:
            self.reiniciar()
            return 0, "SIN CARACTERISTICAS VALIDAS"
        centros_clase = self.centros[self.etiquetas == indice]
        distancia = float(np.clip(1.0 - np.max(centros_clase @ (f / norma)), 0.0, 2.0))
        detalle = (f"{self.clases[indice]} {confianza * 100:.1f}% | "
                   f"distancia {distancia:.4f}/{self.limites[indice]:.4f}")
        if confianza < self.prob_min or distancia > self.limites[indice]:
            self.reiniciar()
            return 0, f"SIN FIGURA ACEPTADA | {detalle}"
        if self.candidata == indice:
            self.contador = min(self.contador + 1, self.repeticiones)
        else:
            self.candidata = indice
            self.contador = 1
        if self.contador < self.repeticiones:
            return 0, f"CONFIRMANDO {self.contador}/{self.repeticiones} | {detalle}"
        return self.estados[indice], f"ACEPTADA | {detalle}"


def ejecutar_vision():
    import cv2
    from tensorflow.lite.python.interpreter import Interpreter

    carpeta = Path(__file__).resolve().parent
    parejas = [(base / "modelo_rescate.tflite", base / "clases_rescate.json")
               for base in (carpeta, carpeta.parent)]
    pareja = next((par for par in parejas if all(p.is_file() for p in par)), None)
    if pareja is None:
        raise FileNotFoundError("Coloca modelo_rescate.tflite y clases_rescate.json juntos en la app.")
    ruta_modelo, ruta_config = pareja
    config = json.loads(ruta_config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("Usa los archivos del cuaderno actualizado: tres clases y filtro de similitud.")
    if hashlib.sha256(ruta_modelo.read_bytes()).hexdigest() != config.get("model_sha256"):
        raise ValueError("El JSON no corresponde a este TFLite. Copia ambos del mismo ZIP.")
    if config.get("input_color") != "RGB" or config.get("input_range") != [0, 255]:
        raise ValueError("Se requiere entrada RGB de 0 a 255.")
    filtro = FiltroFiguras(config)

    interpreter = Interpreter(model_path=str(ruta_modelo))
    interpreter.allocate_tensors()
    entradas = interpreter.get_input_details()
    salidas = {int(s["index"]): s for s in interpreter.get_output_details()}
    if len(entradas) != 1:
        raise ValueError("Se esperaba una entrada de imagen.")
    entrada = entradas[0]
    forma = tuple(int(n) for n in entrada["shape"])
    if forma != (1, 224, 224, 3) or entrada["dtype"] != np.float32:
        raise ValueError(f"Entrada incompatible: {forma}, {entrada['dtype']}")
    indice_prob = int(config["outputs"]["probabilities_index"])
    indice_feat = int(config["outputs"]["features_index"])
    for indice, forma_salida in ((indice_prob, (1, 3)),
                                (indice_feat, (1, filtro.centros.shape[1]))):
        if (indice not in salidas or salidas[indice]["dtype"] != np.float32
                or tuple(salidas[indice]["shape"]) != forma_salida):
            raise ValueError("Las salidas del TFLite no coinciden con el JSON.")

    cap = cv2.VideoCapture(CAMARA)
    try:
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 20)
        print("Vision: tres clases, RGB y normalizacion dentro del modelo.", flush=True)
        print("Rechazo provisional: aun requiere pruebas con escenas sin figura.", flush=True)
        ultimo_log = 0.0

        def loop_vision():
            nonlocal ultimo_log
            ret, frame = cap.read()
            if not ret:
                filtro.reiniciar()
                enviar_estado(0)
                print("Sin imagen de la camara: LEDs apagados si hay comunicacion.", flush=True)
                time.sleep(0.5)
                return
            # Float32 ANTES del resize: evita redondear los pixeles interpolados.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            interpreter.set_tensor(entrada["index"], img[np.newaxis, ...])
            interpreter.invoke()
            probabilidades = interpreter.get_tensor(indice_prob)[0]
            caracteristicas = interpreter.get_tensor(indice_feat)[0]
            estado, motivo = filtro.evaluar(probabilidades, caracteristicas)
            confirmado = enviar_estado(estado)
            if time.monotonic() - ultimo_log >= 1.0:
                print(" | ".join(f"{n}: {float(p) * 100:.1f}%"
                                 for n, p in zip(filtro.clases, probabilidades)), flush=True)
                print(f"{motivo} | LED {NOMBRES_LED[estado]} | "
                      f"{'confirmado' if confirmado else 'SIN CONFIRMACION'}", flush=True)
                ultimo_log = time.monotonic()
            time.sleep(0.05)

        App.run(user_loop=loop_vision)
    finally:
        cap.release()


if __name__ == "__main__":
    try:
        if MODO_PRUEBA:
            print("Prueba: verde, naranja, rojo y apagados, un segundo cada uno.", flush=True)
            App.run(user_loop=loop_prueba)
        else:
            ejecutar_vision()
    except KeyboardInterrupt:
        print("Deteniendo la app...", flush=True)
    finally:
        if estado_confirmado is not None:
            enviar_estado(0, forzar=True)
        # Si Python termina sin poder enviar, el sketch apaga tras 5 s sin ordenes.
