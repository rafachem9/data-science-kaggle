from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('runs/detect/train9/weights/best.pt')  # Cambia 'best.pt' por el archivo del modelo entrenado
nivel_confianza = 0.4

# Hacer inferencia en una imagen
results = model('datasets/others/image_check1.jpg', save=True, imgsz=640, conf=nivel_confianza)
results = model('datasets/others/image_check3.jpg', save=True, imgsz=640, conf=nivel_confianza)
results = model('datasets/others/test_resized.PNG', save=True, imgsz=640, conf=nivel_confianza)