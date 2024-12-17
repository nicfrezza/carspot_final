import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

area_coords = [
   [(1150, 698), (1225, 685), (1326, 768), (1232, 786)]  
]

cap = cv2.VideoCapture(r"C:\Users\nicol\Downloads\carspot\carspot_1-versaofirebase\carspot_1-versaofirebase\yolov8parkingspace-main\yolov8parkingspace-main\parking1.mp4")

with open("C:\\Users\\nicol\\Downloads\\carspot\\carspot_1-versaofirebase\\carspot_1-versaofirebase\\yolov8parkingspace-main\\yolov8parkingspace-main\\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    cars_in_area = []

    for index, row in px.iterrows():
        x1 = int(row[0]) 
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            for i, area in enumerate(area_coords):
                results = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if results >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, f'Carro na Área {i + 1}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cars_in_area.append(c)

    for i, area in enumerate(area_coords):
        if len(cars_in_area) > 0:
            cv2.polylines(frame, [np.array(area, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'Área {i + 1}', (area[0][0] + 10, area[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.polylines(frame, [np.array(area, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f'Área {i + 1}', (area[0][0] + 10, area[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Detecção de Carros nas Áreas", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
