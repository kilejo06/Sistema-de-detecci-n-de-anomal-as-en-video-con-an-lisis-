import cv2
import time
import json
import io
import numpy as np
import redis
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────

CAMERA_ID       = "CAM-01"
RTSP_URL        = 0              # 0 = webcam principal del Mac

TARGET_FPS      = 10         # 16 frames @ 10 fps = 1.6 s por clip
CLIP_LENGTH     = 16         # frames válidos por clip
IMG_SIZE        = 224        # resolución de entrada del modelo

# Umbrales de control de calidad
BLUR_THRESHOLD      = 20.0  # varianza del Laplaciano; por debajo → frame borroso
BRIGHTNESS_MIN      = 30.0   # brillo promedio mínimo (0–255); por debajo → muy oscuro
BRIGHTNESS_MAX      = 240.0  # brillo promedio máximo; por encima → sobreexpuesto
FREEZE_THRESHOLD    = 0.003  # fracción de píxeles diferentes; por debajo → congelado

# Normalización ImageNet (media y desviación estándar por canal BGR→RGB)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Redis
REDIS_HOST      = "localhost"
REDIS_PORT      = 6379
REDIS_STREAM    = "clips:cam01"
REDIS_MAXLEN    = 100        # máximo de clips pendientes en el stream

# Shared Layer (ResNet18 + LSTM)
MODEL_DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
REDIS_CONSUMER_BLOCK_MS = 2000
LSTM_HIDDEN_DIM         = 256
MODO_EJECUCION          = "input"  # "input" o "shared"


# ─────────────────────────────────────────────
# FASE 1 · SUBMUESTREO
# ─────────────────────────────────────────────

def calcular_intervalo_submuestreo(fps_camara: float, fps_objetivo: int) -> int:
    intervalo = max(1, round(fps_camara / fps_objetivo))
    reduccion = (1 - 1 / intervalo) * 100
    print(f"[Submuestreo] {fps_camara:.0f} fps → {fps_objetivo} fps "
          f"(intervalo={intervalo}, reducción={reduccion:.0f}% de carga)")
    return intervalo


# ─────────────────────────────────────────────
# FASE 2 · TIMESTAMP
# ─────────────────────────────────────────────

def obtener_timestamp_ms() -> int:
    return int(time.time() * 1000)


# ─────────────────────────────────────────────
# FASE 3 · PREPROCESAMIENTO: RESIZE + NORMALIZACIÓN
# ─────────────────────────────────────────────

def preprocesar_frame(frame: np.ndarray) -> np.ndarray:
    # Resize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)

    # Normalización ImageNet: (pixel/255 - media) / std por canal
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_norm = (frame_norm - IMAGENET_MEAN) / IMAGENET_STD

    return frame_norm


# ─────────────────────────────────────────────
# FASE 4 · MÁSCARA ROI (Región de Interés)
# ─────────────────────────────────────────────

def crear_mascara_roi(alto: int, ancho: int,
                      poligono: list[tuple[int, int]] | None = None) -> np.ndarray:
    mascara = np.ones((alto, ancho), dtype=np.uint8)

    if poligono is not None:
        mascara = np.zeros((alto, ancho), dtype=np.uint8)
        puntos  = np.array(poligono, dtype=np.int32)
        cv2.fillPoly(mascara, [puntos], 1)
        print(f"[ROI] Máscara aplicada con polígono de {len(poligono)} vértices")
    else:
        print("[ROI] Sin restricción de zona — se usa el frame completo")

    return mascara


def aplicar_mascara_roi(frame: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(frame, frame,
                           mask=cv2.resize(mascara,
                                           (frame.shape[1], frame.shape[0])))


# ─────────────────────────────────────────────
# FASE 5 · CONTROL DE CALIDAD (3 criterios)
# ─────────────────────────────────────────────

def evaluar_borrosidad(frame_gris: np.ndarray) -> tuple[bool, float]:
    score = cv2.Laplacian(frame_gris, cv2.CV_64F).var()
    return score >= BLUR_THRESHOLD, score


def evaluar_brillo(frame_gris: np.ndarray) -> tuple[bool, float]:
    score = float(np.mean(frame_gris))
    return BRIGHTNESS_MIN <= score <= BRIGHTNESS_MAX, score


def evaluar_congelamiento(frame_gris: np.ndarray,
                          frame_anterior_gris: np.ndarray | None) -> tuple[bool, float]:
    if frame_anterior_gris is None:
        return True, 1.0   # primer frame: no hay referencia, se acepta

    diff  = cv2.absdiff(frame_gris, frame_anterior_gris)
    _, mask = cv2.threshold(diff, 10, 1, cv2.THRESH_BINARY)
    fraccion_cambio = float(mask.sum()) / mask.size
    return fraccion_cambio >= FREEZE_THRESHOLD, fraccion_cambio


def controlar_calidad(frame: np.ndarray,
                      frame_anterior: np.ndarray | None) -> dict:
    gris          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ant_gris      = (cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
                     if frame_anterior is not None else None)

    ok_blur,    s_blur    = evaluar_borrosidad(gris)
    ok_bright,  s_bright  = evaluar_brillo(gris)
    ok_freeze,  s_freeze  = evaluar_congelamiento(gris, ant_gris)

    valido = ok_blur and ok_bright and ok_freeze

    if not ok_blur:
        razon = f"borroso (laplaciano={s_blur:.1f} < {BLUR_THRESHOLD})"
    elif not ok_bright:
        razon = f"brillo fuera de rango ({s_bright:.1f})"
    elif not ok_freeze:
        razon = f"señal congelada (cambio={s_freeze:.4f} < {FREEZE_THRESHOLD})"
    else:
        razon = "ok"

    return {
        "valido":  valido,
        "razon":   razon,
        "scores":  {
            "laplaciano": round(s_blur, 2),
            "brillo":     round(s_bright, 2),
            "cambio":     round(s_freeze, 5),
        }
    }


# ─────────────────────────────────────────────
# FASE 6 · BUFFER REDIS
# ─────────────────────────────────────────────

def serializar_clip(frames: list[np.ndarray]) -> bytes:
    """
    Serializa un clip (16 frames normalizados) para enviarlo a Redis.

    Se usa np.save sobre un buffer en memoria para preservar
    el dtype float32 y la forma (16, 224, 224, 3) sin pérdida.
    """
    buf = io.BytesIO()
    array_clip = np.stack(frames, axis=0)          # (16, 224, 224, 3)
    np.save(buf, array_clip)
    return buf.getvalue()


def depositar_en_redis(r: redis.Redis,
                       frames: list[np.ndarray],
                       timestamps: list[int],
                       scores_lista: list[dict]) -> str:
    """
    Deposita un clip completo (16 frames válidos) en el stream Redis.

    Redis Streams desacopla la captura del análisis:
    el Input Layer escribe a su ritmo; la Shared Layer lee a su ritmo.
    Retorna el ID del mensaje generado por Redis.
    """
    payload_binario = serializar_clip(frames)

    metadata = {
        "camera_id":  CAMERA_ID,
        "ts_inicio":  timestamps[0],
        "ts_fin":     timestamps[-1],
        "n_frames":   len(frames),
        "scores":     json.dumps(scores_lista),
    }

    msg_id = r.xadd(
        REDIS_STREAM,
        {"clip": payload_binario, **metadata},
        maxlen=REDIS_MAXLEN
    )
    return msg_id


def calcular_duracion_clip_s(timestamps: list[int]) -> float:
    """
    Duración temporal del clip en segundos.

    Con 16 frames a 10 fps, la cobertura temporal es 1.6 s.
    Para reflejarlo, se suma un intervalo promedio al delta
    entre primer y último timestamp.
    """
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(np.array(timestamps, dtype=np.int64))
    intervalo_ms = float(np.mean(diffs))
    return ((timestamps[-1] - timestamps[0]) + intervalo_ms) / 1000.0


# ─────────────────────────────────────────────
# FASE 7 · SHARED LAYER (RESNET18 + LSTM)
# ─────────────────────────────────────────────

class ResNet18FeatureExtractor(nn.Module):
    """
    Extrae features frame a frame usando ResNet18 preentrenada en ImageNet.

    Entrada : (N, 3, 224, 224)
    Salida  : (N, 512)
    """
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # elimina la cabeza de clasificación

        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TemporalLSTMEncoder(nn.Module):
    """
    Codifica secuencias temporales de features.

    Entrada : (B, T, 512)
    Salida  : (B, hidden_dim)
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = LSTM_HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


def deserializar_clip(payload_binario: bytes) -> np.ndarray:
    """
    Reconstruye el clip desde Redis con forma esperada (16, 224, 224, 3).
    """
    buf = io.BytesIO(payload_binario)
    clip = np.load(buf, allow_pickle=False)

    shape_esperada = (CLIP_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    if clip.shape != shape_esperada:
        raise ValueError(
            f"Forma de clip inválida: {clip.shape}. Esperada: {shape_esperada}"
        )
    if clip.dtype != np.float32:
        clip = clip.astype(np.float32)
    return clip


def clip_numpy_a_tensor(clip: np.ndarray) -> torch.Tensor:
    """
    Convierte de NHWC a NCHW para ResNet.
    Entrada : (16, 224, 224, 3)
    Salida  : (16, 3, 224, 224)
    """
    return torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()


def _obtener_campo(fields: dict, key: str):
    if key in fields:
        return fields[key]
    key_b = key.encode("utf-8")
    if key_b in fields:
        return fields[key_b]
    return None


def leer_siguiente_clip_redis(
    r: redis.Redis,
    last_id: str = "$",
    block_ms: int = REDIS_CONSUMER_BLOCK_MS,
) -> tuple[dict | None, str]:
    """
    Lee un clip nuevo del stream Redis con XREAD.
    """
    mensajes = r.xread({REDIS_STREAM: last_id}, count=1, block=block_ms)
    if not mensajes:
        return None, last_id

    _, entries = mensajes[0]
    msg_id_raw, fields = entries[0]
    clip_binario = _obtener_campo(fields, "clip")
    if clip_binario is None:
        raise KeyError("El mensaje Redis no contiene el campo 'clip'")

    msg_id = msg_id_raw.decode("utf-8") if isinstance(msg_id_raw, bytes) else str(msg_id_raw)
    return {"id": msg_id, "fields": fields, "clip": clip_binario}, msg_id


def procesar_siguiente_clip_redis(
    r: redis.Redis,
    feature_extractor: ResNet18FeatureExtractor,
    lstm_encoder: TemporalLSTMEncoder,
    last_id: str = "$",
    device: str = MODEL_DEVICE,
) -> tuple[dict | None, str]:
    """
    Pipeline:
        clip Redis (16,224,224,3) -> ResNet18 frame a frame -> (16,512)
        -> LSTM temporal -> (256)
    """
    payload, nuevo_last_id = leer_siguiente_clip_redis(r, last_id=last_id)
    if payload is None:
        return None, nuevo_last_id

    clip_np = deserializar_clip(payload["clip"])
    clip_tensor = clip_numpy_a_tensor(clip_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        features_por_frame = feature_extractor(clip_tensor)          # (16, 512)
        secuencia = features_por_frame.unsqueeze(0)                  # (1, 16, 512)
        representacion = lstm_encoder(secuencia).squeeze(0)          # (256,)

    resultado = {
        "msg_id": payload["id"],
        "features_16x512": features_por_frame.cpu().numpy(),
        "embedding_256": representacion.cpu().numpy(),
    }
    return resultado, nuevo_last_id


def ejecutar_shared_layer():
    """
    Shared Layer continuo: consume clips desde Redis y ejecuta ResNet18 + LSTM.
    """
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    device = MODEL_DEVICE
    feature_extractor = ResNet18FeatureExtractor().to(device).eval()
    lstm_encoder = TemporalLSTMEncoder().to(device).eval()
    last_id = "$"

    print(f"\n[SharedLayer] Iniciando consumidor en Redis stream={REDIS_STREAM}")
    print(f"[SharedLayer] Dispositivo: {device}")
    print("[SharedLayer] Pipeline activo: clip (16,224,224,3) -> ResNet18 -> LSTM\n")

    while True:
        resultado, last_id = procesar_siguiente_clip_redis(
            r=r,
            feature_extractor=feature_extractor,
            lstm_encoder=lstm_encoder,
            last_id=last_id,
            device=device,
        )
        if resultado is None:
            continue

        print(
            f"[SharedLayer] Clip ID={resultado['msg_id']} procesado | "
            f"features={resultado['features_16x512'].shape} | "
            f"embedding={resultado['embedding_256'].shape}"
        )


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar_input_layer(rtsp_url: str = RTSP_URL,
                         roi_poligono: list | None = None):
    """
    Pipeline continuo del Input Layer.

    Flujo por iteración:
        1. Captura frame del stream
        2. Submuestreo: descarta frames intermedios
        3. Asigna timestamp
        4. Aplica máscara ROI
        5. Control de calidad (3 criterios)
           ├─ Rechazado → descarta, continúa
           └─ Válido    → preprocesa (resize + norm) → acumula en buffer
        6. Cuando buffer llega a 16 frames → deposita clip en Redis → vacía buffer
    """
    # Conexiones
    cap = cv2.VideoCapture(rtsp_url)
    r   = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el stream: {rtsp_url}")

    fps_camara = cap.get(cv2.CAP_PROP_FPS) or 25.0
    intervalo  = calcular_intervalo_submuestreo(fps_camara, TARGET_FPS)

    ret, primer_frame = cap.read()
    alto, ancho       = primer_frame.shape[:2]
    mascara_roi       = crear_mascara_roi(alto, ancho, roi_poligono)

    # Estado del pipeline
    buffer_frames     = []
    buffer_timestamps = []
    buffer_scores     = []
    frame_anterior    = None
    indice_frame      = 0
    clips_enviados    = 0
    frames_rechazados = 0

    print(f"\n[InputLayer] Iniciando captura — cámara: {CAMERA_ID}")
    print(f"[InputLayer] Stream: {rtsp_url}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[InputLayer] Stream interrumpido — reintentando...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        indice_frame += 1

        # ── FASE 1: Submuestreo ──────────────────────────────────────
        if indice_frame % intervalo != 0:
            continue

        # ── FASE 2: Timestamp ────────────────────────────────────────
        ts = obtener_timestamp_ms()

        # ── FASE 4: Máscara ROI ──────────────────────────────────────
        frame_roi = aplicar_mascara_roi(frame, mascara_roi)

        # ── FASE 5: Control de calidad ───────────────────────────────
        resultado_qc = controlar_calidad(frame_roi, frame_anterior)

        if not resultado_qc["valido"]:
            frames_rechazados += 1
            print(f"  [QC] Frame #{indice_frame} rechazado → {resultado_qc['razon']}")
            continue

        frame_anterior = frame_roi.copy()

        # ── FASE 3: Preprocesamiento (resize + normalización) ────────
        frame_procesado = preprocesar_frame(frame_roi)

        # Acumular en buffer
        buffer_frames.append(frame_procesado)
        buffer_timestamps.append(ts)
        buffer_scores.append(resultado_qc["scores"])

        # ── FASE 6: Depositar clip en Redis ──────────────────────────
        if len(buffer_frames) == CLIP_LENGTH:
            msg_id = depositar_en_redis(r, buffer_frames,
                                        buffer_timestamps, buffer_scores)
            clips_enviados += 1
            duracion_clip = calcular_duracion_clip_s(buffer_timestamps)
            print(f"[Redis] Clip #{clips_enviados} depositado — "
                  f"ID={msg_id} | "
                  f"duración={duracion_clip:.1f}s | "
                  f"rechazados acumulados={frames_rechazados}")

            # Vaciar buffer para el siguiente clip
            buffer_frames.clear()
            buffer_timestamps.clear()
            buffer_scores.clear()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if MODO_EJECUCION == "input":
        # roi_poligono=None → usa el frame completo de la webcam
        ejecutar_input_layer(rtsp_url=RTSP_URL, roi_poligono=None)
    elif MODO_EJECUCION == "shared":
        ejecutar_shared_layer()
    else:
        raise ValueError("MODO_EJECUCION debe ser 'input' o 'shared'")
