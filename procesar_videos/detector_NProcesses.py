# Codigo que realiza el analisis del video usando N cores, de 1 a la cantidad de cores de la CPU disponibles en el dispositivo
# p - Placas               t - Transporte

import cv2
import pytesseract
import numpy as np
from time import time
import multiprocessing
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
    frame, model_t, model_p = frame_data
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    results_t = model_t(frame)[0] # Pasar el frame por el modelo YOLO
    detections_t = sv.Detections.from_ultralytics(results_t) # Convertir resultados para Supervision
    class_id = [2, 3, 5, 7] # Etiquetas: car, motorcycle, bus, truck
    detections_t = detections_t[np.isin(detections_t.class_id, class_id)] # Filtrar detecciones solo para vehículos

    # Procesar si se detectan vehículos
    if len(detections_t) > 0:
        annotated_image_t = bounding_box_annotator.annotate(scene=frame, detections=detections_t)
        annotated_image_t = label_annotator.annotate(scene=annotated_image_t, detections=detections_t)
        cropped_image_t = cropped(detections_t, frame)
        results_p = model_p(cropped_image_t, agnostic_nms=True)[0] # Modelo para detectar matrículas
        detections_p = sv.Detections.from_ultralytics(results_p)

        if len(detections_p) > 0:
            cropped_image_matricula = cropped(detections_p, cropped_image_t)

            # Procesar coordenadas para anotaciones
            dif_x = results_p.boxes.xyxy[0][2] - results_p.boxes.xyxy[0][0]
            dif_y = results_p.boxes.xyxy[0][3] - results_p.boxes.xyxy[0][1]
            x1_nuevo = detections_t.xyxy[0][0] + detections_p.xyxy[0][0]
            y1_nuevo = detections_t.xyxy[0][1] + detections_p.xyxy[0][1]
            x2_nuevo = x1_nuevo + dif_x
            y2_nuevo = y1_nuevo + dif_y
            detections_p.xyxy = np.array([[x1_nuevo, y1_nuevo, x2_nuevo, y2_nuevo]])
            annotated_image_p = bounding_box_annotator.annotate(scene=frame, detections=detections_p)
            annotated_image_p = label_annotator.annotate(scene=annotated_image_p, detections=detections_p)

            # Lectura OCR con Tesseract
            gray = cv2.cvtColor(cropped_image_matricula, cv2.COLOR_BGR2GRAY)
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
            data = pytesseract.image_to_string(
                gray, lang='eng',
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            )
            data = data.strip() # Limpiar la cadena

            # Añadir texto de la matrícula
            text = data
            position = (900, 60) # Posición del texto
            font = cv2.FONT_HERSHEY_SIMPLEX # Tipo de fuente
            font_scale = 2 # Tamaño/escala de la fuente
            font_color = (255, 255, 255) # Color del texto, el formato es (Blue,Green,Red)
            font_thickness = 6 # Grosor de la fuente
            frame = cv2.putText(annotated_image_p, text, position, font, font_scale, font_color, font_thickness)
            # print(text)

    return frame

def main():
    set_start_method("spawn", force=True)
    model_t = YOLO('models\\yolo11n.pt') # Modelo para detectar vehículos
    model_p = YOLO('models\\placa.pt') # Modelo para detectar matrículas
    
    with open("tiempo_ejecucion.txt", "w") as file:
        for num_processes in range(1, multiprocessing.cpu_count() + 1):
            start_time = time()
            cap = cv2.VideoCapture('videos/test.mp4')
            frame_number = 0

            if not cap.isOpened():
                print("Error: No se pudo abrir el video")
                exit()

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del frame
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Largo del frame
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(fps, "FPS")
            pool = Pool(processes=num_processes)
            output_video_path = f'videos/output_video_test_{num_processes}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codificador para el archivo de salida
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read() # Leer el video frame a frame
                frame_number += 1

                if not ret:
                    break

                # print("Número de frame: ", frame_number)
                frame_data = (frame, model_t, model_p)
                result = pool.apply(process_frame, (frame_data,))
                out.write(result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            pool.close()
            pool.join()
            end_time = time()
            execution_time = end_time - start_time
            file.write(f"Procesos: {num_processes}, Tiempo de ejecucion: {execution_time:.2f} segundos\n")
            print(f"Procesos: {num_processes}, Tiempo de ejecución: {execution_time:.2f} segundos")

if __name__ == "__main__":
    main()