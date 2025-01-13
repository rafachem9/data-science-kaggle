from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
import torch

from utils import PROJECT_DIR

# Tamaño de entrada del modelo (debe coincidir con el tamaño de imagen con el que entrenaste el modelo)
input_size = 640  # Asumiendo que usaste 640x640 durante el entrenamiento


# Creating a class for object detection which plots boxes and scores frames in addition to detecting an
# object
model_path = f'{PROJECT_DIR}/datasets/input-test/best.pt'
video_path = f'{PROJECT_DIR}/datasets/input-test/partido-mas-telde-san-isidro.mp4'
# video_path = f'{PROJECT_DIR}/datasets/input-test/grancanariavslanzarote.mp4'

# https://www.kaggle.com/code/nityampareek/using-deepsort-object-tracker-with-yolov5
# https://medium.com/@gayathri.s.de/object-detection-and-tracking-with-yolov8-and-deepsort-5d5981752151

# Cargar el modelo entrenado
model = YOLO(model_path)

# Leer un video
cap = cv2.VideoCapture(video_path)

# Cargar el modelo YOLOv8
model = YOLO(model_path)

# Inicializar DeepSORT
deepsort = DeepSort()

# Leer el video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    start = time.perf_counter()

    # Redimensionar el frame al tamaño de entrada del modelo
    img = cv2.resize(frame, (input_size, input_size))

    if not ret:
        break

    # Realizar inferencia en el cuadro
    results = model.predict(img, conf=0.4)

    # Extraer las coordenadas de las cajas y las confidencias
    boxes = results[0].boxes.xywh
    confidences = results[0].boxes.conf

    # Convertir el tensor completo en una lista

    class_ids = results[0].boxes.cls

    boxes_list = boxes.tolist()
    class_ids_list = class_ids.tolist()
    confidences_list = confidences.tolist()
    detections = []

    x_shape, y_shape = frame.shape[1], frame.shape[0]
    n = len(boxes_list)
    for row in range(n):
        x1, y1, x2, y2 = (int(boxes_list[row][0]),
                          int(boxes_list[row][1]),
                          int(boxes_list[row][2]),
                          int(boxes_list[row][3]))
        coords = [x1, y1, int(x2 - x1), int(y2 - y1)]
        print(coords)
        print(boxes_list[row])
        detections.append((coords, confidences_list[row], class_ids_list[row]))

    img = results[0].plot()

    # Realizar el rastreo con DeepSORT
    tracks = deepsort.update_tracks(detections, frame=img)

    # Dibujar las cajas y las IDs de los jugadores
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 255, 0), 2)

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    # Mostrar el cuadro con las detecciones y el rastreo
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow('Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
