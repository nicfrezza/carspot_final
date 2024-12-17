import cv2
import yt_dlp
import numpy as np
import logging
import os 
from ultralytics import YOLO

class ParkingSpaceMarker:
    def __init__(self, stream_url):
        logging.basicConfig(level=logging.INFO)

        self.stream_url = stream_url
        self.model = YOLO('yolov8s.pt') 
        self.area_coords = []  
        self.current_spot = 0  
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  
        self.is_running = True

    def get_stream_url(self):
        """Extrai a URL do stream ao vivo"""
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
        """Captura os cliques do mouse para as coordenadas das vagas"""
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
        """Desenha um polígono (quadrilátero) para a vaga definida"""
        start_idx = spot_index * 4
        points = np.array(self.area_coords[start_idx:start_idx + 4], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(self.current_frame, [points], isClosed=True, color=self.colors[spot_index], thickness=2)
        cv2.imshow("Stream ao Vivo", self.current_frame)

    def save_coordinates(self):
        """Salva as coordenadas das vagas selecionadas"""
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
        """Abre automaticamente o arquivo onde as coordenadas foram salvas"""
        try:
            if os.name == 'nt': 
                os.startfile("areas_salvas.txt")
            elif os.name == 'posix': 
                os.system("open areas_salvas.txt")
        except Exception as e:
            print(f"Erro ao tentar abrir o arquivo: {e}")

    def run(self):
        """Captura e processa o stream de vídeo"""
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

def main():
    stream_url = "https://www.youtube.com/live/nKQBPEVbKYY?si=VCYeojcudVtoZabT"
    marker = ParkingSpaceMarker(stream_url)
    marker.run()

if __name__ == "__main__":
    main()
