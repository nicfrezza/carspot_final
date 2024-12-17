import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Ponto clicado: {x}, {y}")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Marcação da área", frame)

cv2.namedWindow("Marcação da área")
cv2.setMouseCallback("Marcação da área", click_event)

cap = cv2.VideoCapture(r"C:\Users\nicol\Downloads\carspot\carspot_1-versaofirebase\carspot_1-versaofirebase\yolov8parkingspace-main\yolov8parkingspace-main\parking1.mp4")

ret, frame = cap.read()

if not ret:
    print("Erro ao carregar o vídeo")
    cap.release()
    cv2.destroyAllWindows()

cv2.imshow("Marcação da área", frame)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('s') and len(points) >= 3:
        with open('area_coordenadas.txt', 'w') as f:
            for point in points:
                f.write(f"{point[0]}, {point[1]}\n")
        print("Coordenadas salvas em 'area_coordenadas.txt'")
        break

cap.release()
cv2.destroyAllWindows()
