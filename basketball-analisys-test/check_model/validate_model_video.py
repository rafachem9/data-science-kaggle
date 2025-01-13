from ultralytics import YOLO
import cv2
from utils import PROJECT_DIR_FIX


# Tamaño de entrada del modelo (debe coincidir con el tamaño de imagen con el que entrenaste el modelo)
input_size = 640  # Asumiendo que usaste 640x640 durante el entrenamiento

# Cargar el modelo entrenado
model = YOLO(f'{PROJECT_DIR_FIX}/runs/detect/train9/weights/best.pt')

# Leer un video
cap = cv2.VideoCapture(f'{PROJECT_DIR_FIX}/datasets/input-test/partido-mas-telde-san-isidro.mp4')  # Cambia '0' para usar una webcam


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame al tamaño de entrada del modelo
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Realizar inferencia en cada cuadro
    results = model.predict(frame_resized, conf=0.4)

    # Dibujar las detecciones en el cuadro
    annotated_frame = results[0].plot()

    # Mostrar el cuadro procesado
    cv2.imshow('YOLO Inference', annotated_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
