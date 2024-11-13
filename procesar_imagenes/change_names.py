# Modulo para renombrar las imagenes del dataset original para mayor flexibilidad.

import os
import shutil

ruta_directorio_origen = "images" # Directorio donde se encuentran las imágenes originales
ruta_directorio_destino = "imgs" # Directorio donde se guardarán las imágenes renombradas

if not os.path.exists(ruta_directorio_destino):
    os.makedirs(ruta_directorio_destino)

archivos = os.listdir(ruta_directorio_origen)
contador = 1

# Iterar sobre cada archivo
for nombre_archivo in archivos:
    nuevo_nombre = f"img_{contador}.jpg"
    ruta_original = os.path.join(ruta_directorio_origen, nombre_archivo)
    ruta_nuevo = os.path.join(ruta_directorio_destino, nuevo_nombre)
    shutil.copyfile(ruta_original, ruta_nuevo)
    print(f"El archivo {nombre_archivo} ha sido renombrado y copiado como {nuevo_nombre} en {ruta_directorio_destino}")
    contador += 1
