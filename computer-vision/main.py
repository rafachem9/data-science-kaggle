from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('input_videos/input_basketball.jpeg',save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)