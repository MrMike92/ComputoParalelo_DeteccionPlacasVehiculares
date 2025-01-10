#Intento de optimización del código anterior, cargando los modelos una sola vez antes de procesar los frames, junto con otros pequenos cambios.
import cv2
import pytesseract
import numpy as np
from multiprocessing import Pool, set_start_method
import time as t
from ultralytics import YOLO

def cropped(detection, image): # Recortar imagen
    xmin, ymin, xmax, ymax = map(int, detection[:4])
    return image[ymin:ymax, xmin:xmax]

def process_frame(frame_data):
    frame, model_t, model_p = frame_data # Desempaquetar datos del frame y modelos 
    results_t = model_t(frame)[0] # Detección de vehículos
    detections_t = [det for det in results_t.boxes.xyxy if det[-1] in [2, 3, 5, 7]]  # Etiquetas: car, motorcycle, bus, truck

    if detections_t:
        cropped_image_t = cropped(detections_t[0], frame)  # Considerar solo la primera detección de vehículo
        results_p = model_p(cropped_image_t, agnostic_nms=True)[0]
        detections_p = results_p.boxes.xyxy

        if detections_p:
            cropped_image_plate = cropped(detections_p[0], cropped_image_t)  # Primera matrícula detectada
            gray = cv2.cvtColor(cropped_image_plate, cv2.COLOR_BGR2GRAY)
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
            plate_text = pytesseract.image_to_string(
                gray, lang='eng',
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            ).strip()

            if plate_text:
                position = (50, 50)
                frame = cv2.putText(
                    frame, plate_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
    return frame

def main():
    start_time = t.time()
    set_start_method("spawn", force=True)
    cap = cv2.VideoCapture('procesar_videos/videos/test2.mp4')

    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('out_videos/video_optimizado.mp4', fourcc, fps, (width, height))

    # Cargar modelos una sola vez
    model_path_t = 'procesar_videos\models\placa.pt'
    model_path_p = 'procesar_videos\models\yolo11n.pt'
    model_t = YOLO(model_path_t)  #Antes se estaban cargando los modelos model_t y model_pen cada iteración de procesamiento de frames.
    model_p = YOLO(model_path_p) # Ahora se cargan una sola vez antes de procesar los frames.

    frame_data_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_data_list.append((frame, model_t, model_p))

    cap.release()

    # Procesar frames en paralelo
    with Pool(processes=4) as pool: #Usar pool.apply en lugar de pool.map puede causar cuellos de botella
        # 
        results = pool.map(process_frame, frame_data_list)

    for result in results:
        out.write(result)

    out.release()
    print("Tiempo total: ", t.time() - start_time)

if __name__ == "__main__":
    main()
