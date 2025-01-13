from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from utils import PROJECT_DIR
import time
from utils import PROJECT_DIR
from utils_images.functions_util import get_rectangule_coords

import numpy as np

# Path to the video file
video_path = f'{PROJECT_DIR}/datasets/input-test/partido-mas-telde-san-isidro.mp4'

# Path to the diagram image
diagram_image_path = f'{PROJECT_DIR}/datasets/input-test/basket_court.png'
diagram_image = cv2.imread(diagram_image_path)

# Open the video
cap = cv2.VideoCapture(video_path)

# Court coordinates
court_coords = np.array(
    [[-100, 700],  # Esquina inferior izquierda
     [1500, 650],  # Esquina inferior derecha
     [1100, 450],  # Esquina superior derecha
     [100, 480]],  # Esquina superior izquierda
    np.int32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the Court location
    color = (0, 255, 0)  # Verde
    thickness = 2
    cv2.polylines(frame, [court_coords], isClosed=True, color=color, thickness=thickness)
    # Display the frame with the rectangle
    cv2.imshow("Video with Rectangle", frame)
    time.sleep(1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
