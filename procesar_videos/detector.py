# Codigo que realiza una sola vez el analisis del video sin aplicación de computo paralelo
# p - Placas               t - Transporte

import cv2
import pytesseract
import numpy as np
# from time import time
import supervision as sv
from ultralytics import YOLO

# Función para recortar imágenes
def cropped(detections, image):
    bounding_box = detections.xyxy
    xmin, ymin, xmax, ymax = bounding_box[0] # Extraer las coordenadas de la caja delimitadora
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cropped_image = image[ymin:ymax, xmin:xmax] # Recortar la imagen usando las coordenadas de la caja delimitadora
    
    return cropped_image

def main():
    # start_time = time()
    cap = cv2.VideoCapture('videos/test.mp4')
    frame_number = 0

    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Largo del frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps, "FPS")
    output_video_path = 'videos/output_video_test.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codificador para el archivo de salida
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    model_t = YOLO('models\\yolo11n.pt') # Modelo para detectar vehículos

    while cap.isOpened():
        ret, frame = cap.read() # Leer el video frame a frame
        frame_number += 1 

        if not ret:
            break 

        print("Número de frame: ", frame_number) # Número de frame analizado
        results_t = model_t(frame)[0] # Pasar el frame por el modelo de YOLO
        detections_t = sv.Detections.from_ultralytics(results_t) # Convertir resultados para Supervision
        class_id = [2, 3, 5, 7] # Etiquetas: car, motorcycle, bus, truck
        detections_t = detections_t[np.isin(detections_t.class_id, class_id)] # Filtrar detecciones solo para los vehículos

        if len(detections_t) > 0:
            # Descomentar si se desea ver el recuadro del vehiculo
            """bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image_t = bounding_box_annotator.annotate(scene=frame, detections=detections_t)
            annotated_image_t = label_annotator.annotate(scene=annotated_image_t, detections=detections_t)"""
            cropped_image_t = cropped(detections_t, frame)
            model_p = YOLO('models\\placa.pt') # Modelo para detectar matrículas
            results_p = model_p(cropped_image_t, agnostic_nms=True)[0]
            detections_p = sv.Detections.from_ultralytics(results_p)

            if len(detections_p) > 0:
                cropped_image_matricula = cropped(detections_p, cropped_image_t)

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
                frame = cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness) # Comentar si se desea ver el recuadro de la placa
                # frame = cv2.putText(annotated_image_p, text, position, font, font_scale, font_color, font_thickness) # Descomentar si se desea ver el recuadro de la placa
                # print(text) # Imprimir en consola el texto detectado en ese frame

        out.write(frame)
        # cv2.imshow('Frame', frame) # Mostrar el frame mientras se guarda
        
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