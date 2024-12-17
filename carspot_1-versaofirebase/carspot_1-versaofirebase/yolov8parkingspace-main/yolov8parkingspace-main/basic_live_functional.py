import cv2
import yt_dlp
import streamlink
import numpy as np
import logging
import os
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore

class ParkingSpaceMarker:
    def __init__(self, stream_url):
        logging.basicConfig(level=logging.INFO)
        self.stream_url = stream_url
        self.model = YOLO('yolov8s.pt')
        self.area_coords = []
        self.current_spot = 0
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.is_running = True
        self.current_frame = None

    def get_stream_url(self):
        try:
            with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
                info_dict = ydl.extract_info(self.stream_url, download=False)
                video_formats = [
                    f for f in info_dict['formats']
                    if f.get('height') and f.get('width')
                ]
                if not video_formats:
                    raise ValueError("Nenhum stream de vídeo adequado encontrado")
                best_stream = max(video_formats, key=lambda x: x['height'] * x['width'])
                return best_stream['url']
        except Exception as e:
            logging.error(f"Falha ao obter URL do stream: {e}")
            return None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_spot < 4 and len(self.area_coords) < (self.current_spot + 1) * 4:
                self.area_coords.append((x, y))
                print(f"Ponto selecionado para vaga {self.current_spot + 1}: ({x}, {y})")
                cv2.circle(self.current_frame, (x, y), 3, self.colors[self.current_spot], -1)
                cv2.imshow("Stream ao Vivo", self.current_frame)

                if len(self.area_coords) % 4 == 0:
                    self.draw_polygon(self.current_spot)
                    if len(self.area_coords) % 4 == 0:
                        self.current_spot += 1
                    if self.current_spot == 4:
                        self.save_coordinates()
                        print("Todas as 4 vagas foram marcadas e as coordenadas foram salvas!")
                        self.is_running = False
                        self.open_saved_file()

    def draw_polygon(self, spot_index):
        start_idx = spot_index * 4
        points = np.array(self.area_coords[start_idx:start_idx + 4], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(self.current_frame, [points], isClosed=True, color=self.colors[spot_index], thickness=2)
        cv2.imshow("Stream ao Vivo", self.current_frame)

    def save_coordinates(self):
        try:
            with open("areas_salvas.txt", "w") as file:
                for i in range(4):
                    start_idx = i * 4
                    file.write(f"Vaga {i + 1}:\n")
                    for j in range(4):
                        x, y = self.area_coords[start_idx + j]
                        file.write(f"  Ponto {j + 1}: ({x}, {y})\n")
            print("Coordenadas salvas com sucesso!")
        except Exception as e:
            print(f"Erro ao salvar as coordenadas: {e}")

    def open_saved_file(self):
        try:
            if os.name == 'nt':
                os.startfile("areas_salvas.txt")
            elif os.name == 'posix':
                os.system("open areas_salvas.txt")
        except Exception as e:
            print(f"Erro ao tentar abrir o arquivo: {e}")

    def run(self):
        stream_url = self.get_stream_url()
        if not stream_url:
            print("Não foi possível obter a URL do stream")
            return

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir o stream de vídeo")
            return

        cv2.namedWindow('Stream ao Vivo')
        cv2.setMouseCallback('Stream ao Vivo', self.click_event)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("Falha ao capturar o frame")
                    break

                self.current_frame = cv2.resize(frame, (640, 480))
                cv2.imshow('Stream ao Vivo', self.current_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.is_running = False
        finally:
            cap.release()
            cv2.destroyAllWindows()




class ParkingSpaceDetector:
    def __init__(self, stream_url, coordinate_file="areas_salvas.txt", model_path='yolov8s.pt'):
        self.model = YOLO(model_path)
        self.stream_url = stream_url
        
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate('C:\\Users\\nicol\\Downloads\\carspot\\carspot_1-versaofirebase\\carspot_1-versaofirebase\\yolov8parkingspace-main\\yolov8parkingspace-main\\serviceAccountKey.json')  # Verifique o caminho correto
                firebase_admin.initialize_app(cred)
                print("Firebase inicializado com sucesso")
            
            self.db = firestore.client()
            print("Cliente Firestore criado com sucesso")
            
            test_doc = self.db.collection('vagas').document('dSFhSk2IgbQGULdDgkkO').get()
            if test_doc.exists:
                print("Conexão com Firebase testada com sucesso")
            else:
                print("Aviso: Documento de teste não encontrado")
                
        except Exception as e:
            print(f"Erro na inicialização do Firebase: {e}")
            raise

        self.parking_spaces = self.load_parking_spaces(coordinate_file)
        self.class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                   'hair drier', 'toothbrush']

    def get_stream_url(self):
        try:
            with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
                info_dict = ydl.extract_info(self.stream_url, download=False)
                video_formats = [
                    f for f in info_dict['formats']
                    if f.get('height') and f.get('width')
                ]
                if not video_formats:
                    raise ValueError("Nenhum stream de vídeo adequado encontrado")
                best_stream = max(video_formats, key=lambda x: x['height'] * x['width'])
                return best_stream['url']
        except Exception as e:
            logging.error(f"Falha ao obter URL do stream: {e}")
            return None

    def load_parking_spaces(self, coordinate_file):
        if not os.path.exists(coordinate_file):
            logging.error(f"Coordinate file {coordinate_file} not found")
            return {}
            
        parking_spaces = {}
        try:
            with open(coordinate_file, 'r') as file:
                lines = file.readlines()
                if not lines:
                    logging.error("Coordinate file is empty")
                    return {}
                    
                for i in range(0, len(lines), 5):
                    space_name = lines[i].strip()
                    coords = []
                    for j in range(1, 5):
                        x, y = map(int, lines[i+j].split(': (')[1].split(')')[0].split(', '))
                        coords.append((x, y))
                    parking_spaces[space_name] = coords
            return parking_spaces
        except Exception as e:
            logging.error(f"Error loading coordinates: {e}")
            return {}

    def is_point_inside_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def update_firebase(self, space_name, is_occupied):
        try:
            vaga_numero = space_name.split()[1]  # Gets the number from "Vaga X"
            print(f"\n=== INÍCIO DA ATUALIZAÇÃO ===")
            print(f"Tentando atualizar vaga {vaga_numero}")
            print(f"Status a ser definido: {'ocupada' if is_occupied else 'livre'}")

            # Map each parking space to its corresponding Firebase document ID
            document_mapping = {
                "Vaga 1": "dSFhSk2IgbQGULdDgkkO",
                "Vaga 2": "HY2GwtTgOBvH2l7eEGOD",
                "Vaga 3": "9mi8B6LJmkhhWm6gCSlE",
                "Vaga 1 4": "QnuzabAKP4itUq5LU22L"
            }

            if vaga_numero in document_mapping:
                try:
                    if not self.db:
                        print("Erro: Conexão com Firebase não estabelecida")
                        return

                    doc_ref = self.db.collection('vagas').document(document_mapping[vaga_numero])
                    doc = doc_ref.get()
                    
                    if not doc.exists:
                        print(f"Erro: Documento {document_mapping[vaga_numero]} não encontrado")
                        return

                    update_time = doc_ref.update({
                        'status': is_occupied,
                        'timestamp': firestore.SERVER_TIMESTAMP  
                    })
                    
                    print(f"Atualização realizada em: {update_time}")
                    
                    updated_doc = doc_ref.get()
                    print(f"Status atual no Firebase: {updated_doc.to_dict()}")

                except Exception as firebase_error:
                    print(f"Erro específico do Firebase: {firebase_error}")
                    raise
            else:
                print(f"Erro: Vaga {vaga_numero} não encontrada no mapeamento")

        except Exception as e:
            print(f"\n=== ERRO NA ATUALIZAÇÃO ===")
            print(f"Erro completo: {str(e)}")
            import traceback
            print(f"Stack trace:\n{traceback.format_exc()}")

        finally:
            print("=== FIM DA ATUALIZAÇÃO ===\n")


    def detect_parking_spaces(self, frame):
        try:
            results = self.model.predict(frame)
            detections = results[0].boxes.data.cpu().numpy()

            parking_status = {}

            for space_name, space_coords in self.parking_spaces.items():
                polygon = np.array(space_coords, np.int32)
                cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

                is_occupied = False

                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    
                    if (int(class_id) == 2 and confidence > 0.5):  # 2 is car in COCO dataset
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        if self.is_point_inside_polygon((center_x, center_y), space_coords):
                            is_occupied = True
                            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                            break
                
                parking_status[space_name] = is_occupied
                vaga_number = space_name.split()[1]
                self.update_firebase(space_name, is_occupied)

            return frame, parking_status
        except Exception as e:
            logging.error(f"Error in detect_parking_spaces: {e}")
            return frame, {}

    def run(self):
        video_url = self.get_stream_url()
        
        if not video_url:
            print("Não foi possível recuperar a URL do stream")
            return
        
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print("Erro: Não foi possível abrir o stream de vídeo")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame")
                break
            
            frame = cv2.resize(frame, (640, 480))
            processed_frame, parking_status = self.detect_parking_spaces(frame)

            print("Status das Vagas de Estacionamento:")
            for space, occupied in parking_status.items():
                print(f"{space}: {'Ocupada' if occupied else 'Livre'}")

            cv2.imshow('Detecção de Vagas de Estacionamento', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    stream_url = "https://www.youtube.com/live/nKQBPEVbKYY?si=c54cQNVOp6DKIpbS"
    
    user_input = input("Deseja coletar as coordenadas das vagas? (s/n): ").lower()
    
    if user_input == 's':
        marker = ParkingSpaceMarker(stream_url)
        marker.run()
    
    detector = ParkingSpaceDetector(stream_url)
    detector.run()

if __name__ == "__main__":
    main()