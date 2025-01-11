from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track('datasets/partido-example.png', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)



