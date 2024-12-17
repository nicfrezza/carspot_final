import cv2 #processamento de imagens
import yt_dlp #youtube
import numpy as np #calculo numero
import logging #depuramento log
import os # arquivos txt
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore

#marcação das coordenadas
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
                logging.info(f"Stream URL obtido: {best_stream['url']}")
                return best_stream['url']
        except Exception as e:
            logging.error(f"Falha ao obter URL do stream: {e}")
            return None
    #marcação por meio dos cliques
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_spot < 4 and len(self.area_coords) < (self.current_spot + 1) * 4:
                self.area_coords.append((x, y))
                logging.info(f"Ponto selecionado para vaga {self.current_spot + 1}: ({x}, {y})")
                cv2.circle(self.current_frame, (x, y), 3, self.colors[self.current_spot], -1)

                cv2.putText(self.current_frame, f"Marcando: Vaga {self.current_spot + 1}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Stream ao Vivo", self.current_frame)

                if len(self.area_coords) % 4 == 0:
                    self.draw_polygon(self.current_spot)
                    self.current_spot += 1
                    if self.current_spot == 4:
                        self.save_coordinates()
                        logging.info("Todas as 4 vagas foram marcadas e as coordenadas foram salvas!")
                        self.is_running = False
                        self.open_saved_file()
    #marcar poligno
    def draw_polygon(self, spot_index):
        start_idx = spot_index * 4
        points = np.array(self.area_coords[start_idx:start_idx + 4], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(self.current_frame, [points], isClosed=True, color=self.colors[spot_index], thickness=2)

        text_position = np.mean(points, axis=0).astype(int).flatten()  
        cv2.putText(self.current_frame, f"Vaga {spot_index + 1}", tuple(text_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[spot_index], 2, cv2.LINE_AA)

        cv2.imshow("Stream ao Vivo", self.current_frame)
    
    #salvar coordenadas txt
    def save_coordinates(self):
        try:
            with open("areas_salvas.txt", "w") as file:
                for i in range(4):
                    start_idx = i * 4
                    file.write(f"Vaga {i + 1}:\n")
                    for j in range(4):
                        x, y = self.area_coords[start_idx + j]
                        file.write(f"  Ponto {j + 1}: ({x}, {y})\n")
            logging.info("Coordenadas salvas com sucesso!")
        except Exception as e:
            logging.error(f"Erro ao salvar as coordenadas: {e}")
    

    #coordenadas txt
    def open_saved_file(self):
        try:
            if os.name == 'nt':
                os.startfile("areas_salvas.txt")
            elif os.name == 'posix':
                os.system("open areas_salvas.txt")
        except Exception as e:
            logging.error(f"Erro ao tentar abrir o arquivo: {e}")

    #receber url
    def run(self):
        stream_url = self.get_stream_url()
        if not stream_url:
            logging.error("Não foi possível obter a URL do stream")
            return

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logging.error("Erro: Não foi possível abrir o stream de vídeo")
            return

        cv2.namedWindow('Stream ao Vivo')
        cv2.setMouseCallback('Stream ao Vivo', self.click_event)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Falha ao capturar o frame")
                    break

                self.current_frame = cv2.resize(frame, (640, 480))

                cv2.putText(self.current_frame, "Clique para marcar os pontos da vaga", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Stream ao Vivo', self.current_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    logging.info("Processo interrompido pelo usuário")
                    self.is_running = False
        finally:
            cap.release()
            cv2.destroyAllWindows()

#identificação vagas
class ParkingSpaceDetector:
    def __init__(self, stream_url, coordinate_file="areas_salvas.txt", model_path='yolov8s.pt'):
        logging.basicConfig(level=logging.INFO)
        self.model = YOLO(model_path)
        self.stream_url = stream_url
        self.coordinate_file = coordinate_file

        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate('C:\\Users\\nicol\\Downloads\\carspot\\carspot_1-versaofirebase\\carspot_1-versaofirebase\\yolov8parkingspace-main\\yolov8parkingspace-main\\serviceAccountKey.json')  # Verifique o caminho correto
                firebase_admin.initialize_app(cred)
                logging.info("Firebase inicializado com sucesso")
            self.db = firestore.client()
        except Exception as e:
            logging.error(f"Erro ao inicializar o Firebase: {e}")
            raise

        self.parking_spaces = self.load_parking_spaces()

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

    def load_parking_spaces(self):
        if not os.path.exists(self.coordinate_file):
            logging.error(f"Arquivo {self.coordinate_file} não encontrado")
            return {}

        parking_spaces = {}
        try:
            with open(self.coordinate_file, 'r') as file:
                lines = file.readlines()
                for i in range(0, len(lines), 5):
                    space_name = lines[i].strip()
                    coords = []
                    for j in range(1, 5):
                        x, y = map(int, lines[i + j].split(': (')[1].strip(')\n').split(', '))
                        coords.append((x, y))
                    parking_spaces[space_name] = coords
            logging.info("Coordenadas carregadas com sucesso!")
            return parking_spaces
        except Exception as e:
            logging.error(f"Erro ao carregar as coordenadas: {e}")
            return {}

    def detect_parking_spaces(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()

        parking_status = {}
        for space_name, space_coords in self.parking_spaces.items():
            polygon = np.array(space_coords, np.int32)
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            is_occupied = False
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if int(class_id) == 2 and confidence > 0.5:
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    is_occupied = cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0
                    if is_occupied:
                        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
                        break
            parking_status[space_name] = is_occupied
            self.update_firebase(space_name, is_occupied)
        return frame, parking_status


    #atualizar firebase
    def update_firebase(self, space_name, is_occupied):
        try:
            doc_ref = self.db.collection('vagas').document(space_name)
            data = {
                'automovel': 'carro',
                'descricao': space_name,
                'endereco': 'Logan',
                'id': int(space_name.split()[-1].replace(':', '')), 
                'latitude': '41.741651',
                'longitude': '-111.857096',
                'numero': '950',
                'rua': 'Logan',
                'status': 'ocupada' if is_occupied else 'disponivel',
                'tipo': 1
            }
            doc_ref.set(data)
            logging.info(f"Vaga {space_name} atualizada com sucesso no Firebase.")
        except Exception as e:
            logging.error(f"Erro ao atualizar Firebase para a vaga {space_name}: {e}")

    def run(self):
        video_url = self.get_stream_url()
        if not video_url:
            return

        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            logging.error("Erro ao abrir o stream de vídeo")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            processed_frame, _ = self.detect_parking_spaces(frame)

            cv2.imshow('Detecção de Vagas', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    stream_url = "https://www.youtube.com/live/nKQBPEVbKYY?si=c8vB9ho-hBsi8kHE"
    user_input = input("Deseja coletar as coordenadas das vagas? (s/n): ").lower()
    if user_input == 's':
        marker = ParkingSpaceMarker(stream_url)
        marker.run()

    detector = ParkingSpaceDetector(stream_url)
    detector.run()


if __name__ == "__main__":
    main()