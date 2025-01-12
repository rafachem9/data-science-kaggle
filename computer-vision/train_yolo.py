from ultralytics import YOLO

# Crear y entrenar el modelo
# Parece que con el modelo L y XL no se puede entrenar, solo con el modelo S y M. He probado m con pocos epocs y no detecta nada.
model = YOLO('yolov8n.yaml')  # Modelo medium

model.train(data='datasets/data.yaml', epochs=100, imgsz=640)
