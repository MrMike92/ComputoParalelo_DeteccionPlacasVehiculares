# LP - License Plates               V - Vehicle

import cv2
import pytesseract
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Recortar imagenes
def cropped(detections,image):
    bounding_box = detections.xyxy
    xmin, ymin, xmax, ymax = bounding_box[0] # Extraer las coordenadas de la caja delimitadora
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax) # Asegurar de que las coordenadas sean enteros
    cropped_image = image[ymin:ymax, xmin:xmax] # Recortar la imagen usando las coordenadas de la caja delimitadora
    return cropped_image

def main():
    # Parte 1: Procesar el video/dividirlo en frames
    cap = cv2.VideoCapture('videos/test3.mp4') # Cambiar por el la ubicación del archivo a anilzar
    frame_number = 0

    if not cap.isOpened():
       print("Error: No se puede abrir el archivo de video")
       exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps, "FPS")
    output = 'videos/output_video_test3_preentrenadoV2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificador para el archivo de salida
    out = cv2.VideoWriter('videos/output_video_test3_preentrenadoV2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret,frame = cap.read() # Leer el video frame a frame
        frame_number +=1 

        if not ret:
           break 

        model_V = YOLO('models\\yolo11n.pt') # Modelo para detectar los vehiculos
        print("Numero de frame: ",frame_number) # Número de frame que se está analizando
        results_V = model_V(frame)[0] # Pasar el frame por el modelo que detecta los vehiculos
        detection_V = sv.Detections.from_ultralytics(results_V) # Pasar los resultados a supervison
        class_id = [2, 3, 5, 7] # Etiquetas: car, motorcycle, bus and truck (automovil, motocicleta, autobús y camión)

        # Filtrar y mostrar solo las detecciones de 
        if detection_V.class_id[0] in class_id:
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_image_V = bounding_box_annotator.annotate(scene=frame, detections=detection_V)
            annotated_image_V = label_annotator.annotate(scene=annotated_image_V, detections=detection_V)            
            cropped_image_V = cropped(detection_V, frame) # Recortar de imagen de vehiculo
            model_LP = YOLO('models\\placa.pt') # Modelo para detectar el medio de matricula
            results_LP = model_LP(cropped_image_V, agnostic_nms = True)[0] # Se pasa la imagen recortada por el modelo que detecta matriculas
            results_LP.names[0] = "Matricula"
            detections_LP = sv.Detections.from_ultralytics(results_LP) # Pasar los resultados a la libreria supervison
            cropped_image_LP = cropped(detections_LP, cropped_image_V) # Obtener la imagen de la matricula antes de cambiar 

            # Parte 2: Pasar las coordenadas de la matricula a la imagen recortada del vehiculo

            # Diferencia de las coordenadas XY, el largo y ancho de la caja de la matricula
            dif_x = results_LP.boxes.xyxy[0][2] - results_LP.boxes.xyxy[0][0]
            dif_y = results_LP.boxes.xyxy[0][3] - results_LP.boxes.xyxy[0][1]
            #Puntos iniciales, suma del punto de deteccion de la placa mas el del vehiculo
            x1_nuevo = detection_V.xyxy[0][0] + detections_LP.xyxy[0][0]
            y1_nuevo = detection_V.xyxy[0][1] + detections_LP.xyxy[0][1] 
            # Sumar del punto inicial mas las dimensiones de la box de matricula.
            x2_nuevo = x1_nuevo + dif_x  
            y2_nuevo = y1_nuevo + dif_y 
            # Guardar las nuevas coordenadas 
            detections_LP.xyxy = np.array([[x1_nuevo,y1_nuevo,x2_nuevo,y2_nuevo]])
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            # Pasar la informacion que se mostrara en las etiqutes
            annotated_image_LP = bounding_box_annotator.annotate(scene=frame, detections=detections_LP)
            annotated_image_LP = label_annotator.annotate(scene=annotated_image_LP, detections=detections_LP)
        
            # Parte 3: Lectura de la placa con tesseract OCR

            gray = cv2.cvtColor(cropped_image_LP, cv2.COLOR_BGR2GRAY) # Transformar a escala de grises
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

            # Pasar el OCR por la imagen en escala de grises, que filtra solo numero y letras
            data = pytesseract.image_to_string(gray, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYXabcdefghijklmnopqrstuvwxyz')

            # Limpieza de la cadena de salida del OCR
            valor_medio = round(len(data)/2)
            data = data[valor_medio-3:valor_medio+4]
            
            # Parte 4: agregar la martricula en la imagen

            text = data
            position = (900, 60)  # Definir la posición del texto (se puede ajustar)
            font = cv2.FONT_HERSHEY_SIMPLEX # Fuente del texto
            font_scale = 2 # Tamaño del texto
            font_color = (255, 255, 255)  # Color del texto en BGR
            font_thickness = 6 # Grosor de las letras
            frame = cv2.putText(annotated_image_LP, text, position, font, font_scale, font_color, font_thickness) # Añadir el texto a la imagen
            print(text)
            out.write(frame) # Guardar el frame para reconstruir el video
            # cv2.imshow('Frame', frame) # Mostrar el frame mientra se guardan para el video
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

    cap.release()
    out.release()
   
if __name__ == "__main__":
    main()