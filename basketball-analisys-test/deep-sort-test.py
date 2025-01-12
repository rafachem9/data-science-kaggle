from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model_path = f'D:/Users/rafa_/pycharm-project-datos/basketball-analisys-test/runs/detect/train9/weights/best.pt'
video_path = 'datasets/input-test/partido-mas-telde-san-isidro.mp4'

# Cargar el modelo YOLOv8
model = YOLO(model_path)

# Inicializar DeepSORT
deepsort = DeepSort()

# Leer el video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar inferencia en el cuadro
    results = model.predict(frame, conf=0.4)

    # Extraer las coordenadas de las cajas y las confidencias
    boxes = results[0].boxes.xywh
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    # Filtrar solo jugadores ofensivos (suponiendo que 'offensive_player' es class_id 0)
    boxes = boxes[class_ids == 0]
    confidences = confidences[class_ids == 0]

    # Realizar el rastreo con DeepSORT
    tracks = deepsort.update(boxes)

    # Dibujar las cajas y las IDs de los jugadores
    for track in tracks:
        x1, y1, x2, y2, track_id = track[:5]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el cuadro con las detecciones y el rastreo
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
