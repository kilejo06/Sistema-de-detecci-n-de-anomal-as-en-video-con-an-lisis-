import redis
import numpy as np

from main import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_STREAM,
    MODEL_DEVICE,
    ResNet18FeatureExtractor,
    TemporalLSTMEncoder,
    deserializar_clip,
    clip_numpy_a_tensor,
)

import torch


def _obtener_campo(fields: dict, key: str):
    if key in fields:
        return fields[key]
    key_b = key.encode("utf-8")
    if key_b in fields:
        return fields[key_b]
    return None


def leer_ultimo_clip(r: redis.Redis) -> tuple[str, bytes]:
    mensajes = r.xrevrange(REDIS_STREAM, count=1)
    if not mensajes:
        raise RuntimeError(
            f"No hay clips en Redis stream '{REDIS_STREAM}'. "
            "Ejecuta primero el Input Layer."
        )

    msg_id_raw, fields = mensajes[0]
    clip_binario = _obtener_campo(fields, "clip")
    if clip_binario is None:
        raise KeyError("El mensaje Redis no contiene el campo 'clip'")

    msg_id = msg_id_raw.decode("utf-8") if isinstance(msg_id_raw, bytes) else str(msg_id_raw)
    return msg_id, clip_binario


def formatear_vector(nombre: str, vector: np.ndarray, n: int = 16):
    valores = ", ".join(f"{x:.6f}" for x in vector[:n])
    print(f"{nombre} (primeros {n} valores): [{valores}]")


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    msg_id, clip_binario = leer_ultimo_clip(r)

    device = MODEL_DEVICE
    feature_extractor = ResNet18FeatureExtractor().to(device).eval()
    lstm_encoder = TemporalLSTMEncoder().to(device).eval()

    clip_np = deserializar_clip(clip_binario)
    clip_tensor = clip_numpy_a_tensor(clip_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        features_por_frame = feature_extractor(clip_tensor).cpu().numpy()  # (16, 512)
        embedding = (
            lstm_encoder(torch.from_numpy(features_por_frame).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
            .numpy()
        )

    print(f"\n[Evidencia] Clip procesado: {msg_id}")
    print(f"[Evidencia] Forma de features: {features_por_frame.shape}")
    print(f"[Evidencia] Forma de embedding: {embedding.shape}\n")

    formatear_vector("Feature del frame 1", features_por_frame[0], n=16)
    formatear_vector("Feature del frame 2", features_por_frame[1], n=16)
    formatear_vector("Embedding temporal", embedding, n=16)

    print("\n[Evidencia] Submatriz features[0:3, 0:8]:")
    print(np.array2string(features_por_frame[:3, :8], precision=6, suppress_small=False))


if __name__ == "__main__":
    main()
