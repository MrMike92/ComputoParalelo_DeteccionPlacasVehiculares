# Codigo que realiza una sola vez el analisis del video usando N cores
# p - Placas               t - Transporte

import cv2
import pytesseract
import numpy as np
# from time import time
import supervision as sv
from ultralytics import YOLO
from multiprocessing import Pool, set_start_method

# Función para recortar imágenes
def cropped(detections, image):
    bounding_box = detections.xyxy
    xmin, ymin, xmax, ymax = bounding_box[0] # Extraer las coordenadas de la caja delimitadora
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cropped_image = image[ymin:ymax, xmin:xmax] # Recortar la imagen usando las coordenadas de la caja delimitadora
    
    return cropped_image

# Función para procesar un frame
def process_frame(frame_data):
    frame, model_path_t, model_path_p = frame_data
    model_t = YOLO(model_path_t) # Cargar modelo de vehículos
    model_p = YOLO(model_path_p) # Cargar modelo de matrículas
    results_t = model_t(frame)[0] # Pasar el frame por el modelo YOLO
    detections_t = sv.Detections.from_ultralytics(results_t) # Convertir resultados para Supervision
    class_id = [2, 3, 5, 7]  # Etiquetas: car, motorcycle, bus, truck
    detections_t = detections_t[np.isin(detections_t.class_id, class_id)] # Filtrar detecciones solo para vehículos

    if len(detections_t) > 0:
        cropped_image_t = cropped(detections_t, frame)
        results_p = model_p(cropped_image_t, agnostic_nms=True)[0] # Modelo para detectar matrículas
        detections_p = sv.Detections.from_ultralytics(results_p)

        if len(detections_p) > 0:
            cropped_image_matricula = cropped(detections_p, cropped_image_t)
            
            # Lectura OCR con Tesseract
            gray = cv2.cvtColor(cropped_image_matricula, cv2.COLOR_BGR2GRAY)
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
            data = pytesseract.image_to_string(
                gray, lang='eng',
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            ).strip()
            data = data.strip() # Limpiar la cadena
            
            # Añadir texto de la matrícula
            text = data
            position = (900, 60) # Posición del texto
            font = cv2.FONT_HERSHEY_SIMPLEX # Tipo de fuente
            font_scale = 2 # Tamaño/escala de la fuente
            font_color = (255, 255, 255) # Color del texto, el formato es (Blue,Green,Red)
            font_thickness = 6 # Grosor de la fuente
            frame = cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)

    return frame

def main():
    # start_time = time()
    set_start_method("spawn", force=True)
    num_processes = 4 # Cambiar si se desea
    cap = cv2.VideoCapture('videos/test.mp4')
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Largo del frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps, "FPS")
    output_video_path = 'videos/output_video_test_paralelo.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codificador para el archivo de salida
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    model_path_t = 'models\\yolo11n.pt' # Modelo para detectar matrículas
    model_path_p = 'models\\placa.pt' # Modelo para detectar vehículos

    with Pool(processes=num_processes) as pool:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            print("Número de frame: ", frame_number)
            frame_data = (frame, model_path_t, model_path_p)
            result = pool.apply(process_frame, (frame_data,))
            out.write(result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    """end_time = time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")"""

if __name__ == "__main__":
    main()