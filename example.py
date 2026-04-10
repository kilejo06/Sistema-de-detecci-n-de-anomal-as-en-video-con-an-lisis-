import redis, json

r = redis.Redis(host="localhost", port=6379)
mensajes = r.xrevrange("clips:cam01", "+", "-", count=1)

datos = mensajes[0][1]
print("ID del mensaje:", mensajes[0][0])
print("Cámara:        ", datos[b"camera_id"])
print("Frames:        ", datos[b"n_frames"])
print("ts_inicio:     ", datos[b"ts_inicio"])
print("ts_fin:        ", datos[b"ts_fin"])
print("Clip :", datos[b"clip"][:20])

scores = json.loads(datos[b"scores"])
print("Scores (frames 1-3):")
for i, s in enumerate(scores[:2]):
    print(f"  Frame {i+1}: {s}")