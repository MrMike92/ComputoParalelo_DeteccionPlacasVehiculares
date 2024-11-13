# Modulo para preprocesar las iamgenes y recortarlas.

import os
import cv2
from threading import Thread, Lock
import time

cv2.setNumThreads(1)  # Asegurar que cv2 no use más de un hilo
lock = Lock()

def distribuir_carga_techo(n_cores, tamaño_datos):
    tamaño_parte = tamaño_datos // n_cores
    resto = tamaño_datos % n_cores
    rangos = []
    inicio = 0

    for i in range(n_cores):
        fin = inicio + tamaño_parte + (1 if i < resto else 0)
        rangos.append((inicio, fin))
        inicio = fin

    return rangos

def detect_and_crop_plate(image_path, output_folder):  # Función para detectar y recortar la placa
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return False
    
    # Preprocesamiento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)  # Aplicar umbralización binaria inversa
    
    # Encontrar contornos en la imagen umbralizada para identificar la placa
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    plate = None
    
    # Iterar sobre los contornos para identificar la placa
    for contour in contours: 
        x, y, w, h = cv2.boundingRect(contour)  # Obtener las coordenadas y dimensiones del contorno 
        aspect_ratio = w / float(h)  # Calcular la relación de aspecto del contorno
        area = cv2.contourArea(contour)  # Calcular el área del contorno   
        
        # Verificación de proporciones y área para identificar la placa
        if 2 < aspect_ratio < 10 and 10000 < area < 90000:
            plate = image[y:y + h, x:x + w]  # Recortar la región de la placa
            save_path = os.path.join(output_folder, os.path.basename(image_path))  # Ruta de guardado
            
            # Exclusión mutua
            with lock:
                cv2.imwrite(save_path, plate)  # Guardar la imagen recortada en la carpeta de salida
            return True  # Indica que la placa fue encontrada y guardada
    return False  # No se encontró ninguna placa que cumpliera con los criterios


def process_images_in_chunk(image_paths, output_folder):
    # Procesar cada imagen en el grupo
    for image_path in image_paths:
        detect_and_crop_plate(image_path, output_folder)  # Detectar y recortar la placa de la imagen


# Función de procesamiento paralelo con hilos
def parallel_processing(folder_path, output_folder, num_threads):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.jpg')]
    
    # Dividir las imágenes en grupos usando la función de distribución de carga
    rangos = distribuir_carga_techo(num_threads, len(image_paths))
    chunks = [image_paths[inicio:fin] for inicio, fin in rangos]
    
    start_time = time.time()
    
    threads = []
    for chunk in chunks:
        thread = Thread(target=process_images_in_chunk, args=(chunk, output_folder))
        thread.start()  # Iniciar el hilo
        threads.append(thread)
    
    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"Tiempo total de procesamiento: {end_time - start_time:.2f} segundos")

folder_path = 'imgs'
output_folder = 'cropped_imgs'
num_threads = 5
parallel_processing(folder_path, output_folder, num_threads)
