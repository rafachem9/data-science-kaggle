from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
from utils import PROJECT_DIR
from utils_images.functions_util import get_rectangule_coords
import numpy as np

# Setup
model_path = f'{PROJECT_DIR}/datasets/input-test/best.pt'
video_path = f'{PROJECT_DIR}/datasets/input-test/partido-mas-telde-san-isidro.mp4'
# video_path = f'{PROJECT_DIR}/datasets/input-test/nba-video.mp4'
# video_path = f'{PROJECT_DIR}/datasets/input-test/grancanariavslanzarote.mp4'
diagram_image_path = f'{PROJECT_DIR}/datasets/input-test/basket_court.png'

objects_names = ['Ball', 'Hoop', 'Period', 'Player', 'Ref', 'Shot Clock', 'Team Name', 'Team Points', 'Time Remaining']
confidence_level = 0.6

# Court coordinates
court_coords = np.array(
    [[-100, 700],  # Esquina inferior izquierda
     [1500, 650],  # Esquina inferior derecha
     [1100, 450],  # Esquina superior derecha
     [100, 480]],  # Esquina superior izquierda
    np.int32)
diagram_points = np.array([[22, 22], [642, 22], [642, 388], [22, 388]])
matrix_transformation, status = cv2.findHomography(court_coords, diagram_points)

# Cargar el modelo y DeepSORT
model = YOLO(model_path)
deepsort = DeepSort()
deepsort_ref = DeepSort()
deepsort_ball = DeepSort()

# Leer el video
cap = cv2.VideoCapture(video_path)


def get_objects(yogo_results, confidence_level, objects_names, class_name):

    boxes = yogo_results[0].boxes.xywh.tolist()
    confidences = yogo_results[0].boxes.conf.tolist()
    class_ids = yogo_results[0].boxes.cls.tolist()
    detections = []

    for i, box in enumerate(boxes):
        x_center, y_center, w, h = box

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)

        box_update = [x1, y1, int(w), int(h)]
        object_name = objects_names[int(class_ids[i])]

        if confidences[i] > confidence_level:
            if object_name == class_name:
                print(box_update)
                detections.append((box_update, confidences[i], object_name))
    return detections


def plot_result(img, tracks, class_name, class_color, schema_image=None, matrix_transformation=None):
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), class_color, 2)
        cv2.putText(img, f'{class_name}: {track_id}', (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

        if schema_image is not None:
            # Coordenadas del centro del rectángulo
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)

            # Add the center point to the diagram
            center_point = np.array([[[x_center, y_center]]], dtype=np.float32)
            transformed_center = cv2.perspectiveTransform(center_point, matrix_transformation)
            transformed_center = transformed_center[0][0]
            transformed_center = tuple(map(int, transformed_center))

            cv2.circle(schema_image, transformed_center, 5, class_color, -1)
            cv2.putText(schema_image, f'{class_name}: {track_id}',
                        (int(transformed_center[0] - 25), int(transformed_center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
    return img


def main():
    while cap.isOpened():
        ret, frame = cap.read()
        start = time.perf_counter()
        if not ret:
            break

        # Path to the diagram image
        diagram_image = cv2.imread(diagram_image_path)

        # Draw the Court location
        color = (0, 255, 0)  # Verde
        thickness = 2
        cv2.polylines(frame, [court_coords], isClosed=True, color=color, thickness=thickness)

        # Predict with YOLO
        results = model.predict(frame, conf=confidence_level)

        # PLot detections on the frame
        detect_img = results[0].plot()
        cv2.imshow('detect', detect_img)

        # get player detections
        detections_player = get_objects(results, confidence_level, objects_names, 'Player')
        detections_referee = get_objects(results, confidence_level, objects_names, 'Ref')
        detections_ball = get_objects(results, confidence_level, objects_names, 'ball')

        # Follow the players
        tracks_player = deepsort.update_tracks(detections_player, frame=frame)
        frame = plot_result(frame, tracks_player, 'Player',
                            class_color=(255, 0, 0),
                            schema_image=diagram_image,
                            matrix_transformation=matrix_transformation
                            )

        # Follow the referee
        tracks_referee = deepsort_ref.update_tracks(detections_referee, frame=frame)
        frame = plot_result(frame, tracks_referee, 'Ref',
                            class_color=(128, 128, 128),
                            schema_image=diagram_image,
                            matrix_transformation=matrix_transformation)

        # Follow the ball
        tracks_ball = deepsort_ball.update_tracks(detections_ball, frame=frame)
        frame = plot_result(frame, tracks_ball, 'Ball',
                            class_color=(0, 255, 0),
                            schema_image=diagram_image,
                            matrix_transformation=matrix_transformation)

        # Show FPS
        end = time.perf_counter()
        fps = 1 / (end - start)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)

        cv2.imshow("Diagrama", diagram_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

