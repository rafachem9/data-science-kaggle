from ultralytics import YOLO

# Crear y entrenar el modelo
model = YOLO('yolov8x')

model.train(data='datasets/data.yaml', epochs=3, imgsz=640)
