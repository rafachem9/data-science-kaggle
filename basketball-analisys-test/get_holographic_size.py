import cv2
import numpy as np
from utils import PROJECT_DIR
from utils_images.functions_util import get_rectangule_coords

# Crear una imagen negra (500x500 píxeles)
image = np.zeros((720, 1280, 3), dtype="uint8")

# abrir una imagen
diagram_image_path = f'{PROJECT_DIR}/datasets/input-test/basket_court.png'
diagram_image = cv2.imread(diagram_image_path)
print(diagram_image.shape)
height, width, other = image.shape

# abrir una imagen
video_path = f'{PROJECT_DIR}/datasets/input-test/holograpic_test.png'
image = cv2.imread(video_path)
print(image.shape)
# Coordenadas del trapecio
trap_coords = np.array(
    [[-100, 700],  # Esquina inferior izquierda
     [1500, 650],  # Esquina inferior derecha
     [1100, 450],  # Esquina superior derecha
     [100, 480]],  # Esquina superior izquierda
    np.int32)

# Dibujar el contorno del trapecio
color = (0, 255, 0)  # Verde
thickness = 2
cv2.polylines(image, [trap_coords], isClosed=True, color=color, thickness=thickness)

diagram_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
homography_matrix, status = cv2.findHomography(trap_coords, diagram_points)

# Player locations in the video
# player_coords = np.array([[200, 200]], dtype='float32')
# player_coords = np.array([player_coords])


# [85, 557, 55, 153]
# [487, 484, 45, 119]
# [511, 420, 45, 79]
player_coords = [85, 557, 55, 153]
x1, y1, w, h = player_coords
center_x = x1 + w // 2
center_y = y1 + h // 2

cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

# Crear un punto para el centro del jugador
center_point = np.array([[[center_x, center_y]]], dtype=np.float32)  # Debe ser de forma (1, 1, 2)

# Apply the homography
transformed_center = cv2.perspectiveTransform(center_point, homography_matrix)
transformed_center = tuple(map(int, transformed_center[0][0]))

cv2.circle(diagram_image, (50, 50), 5, (0, 0, 255), -1)
cv2.circle(diagram_image, transformed_center, 5, (0, 0, 255), -1)

# Puntos del rectángulo
x2, y2 = x1 + w, y1 + h
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde para el original

cv2.imshow("Rectángulo Dibujado", homography_matrix)

# Mostrar la imagen con el rectángulo
cv2.imshow("Partido", image)

# Mostrar la imagen con el rectángulo
cv2.imshow("Diagrama", diagram_image)

# Esperar hasta que se presione una tecla
cv2.waitKey(0)
