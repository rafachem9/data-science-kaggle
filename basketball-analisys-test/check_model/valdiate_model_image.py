from ultralytics import YOLO
from utils import PROJECT_DIR_FIX

# Cargar el modelo entrenado

model_path = f'{PROJECT_DIR_FIX}/datasets/input-test/best.pt'

model = YOLO(model_path)

nivel_confianza = 0.5

# Hacer inferencia en una imagen
results = model(f'{PROJECT_DIR_FIX}/datasets/input-test/image-test-1.png',
                save=True,
                imgsz=640,
                conf=nivel_confianza)
