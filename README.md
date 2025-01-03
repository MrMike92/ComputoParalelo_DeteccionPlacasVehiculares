# Detección de placas vehiculares.
Un detector de placas vehiculares la cual usa paralelismo para la reducción de tiempo.

## Instrucciones de uso.

- Clona este repositorio.
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python:
    - Las versiones que soporta OpenCV son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta numpy son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta ultralytics son 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta supervision son 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta pytesseract son 3.8, 3.9, 3.10, 3.11, 3.12
- Instalar las siguentes bibliotecas:
```python
# Copiar y pegar lo siguente en el CMD de Windows.
pip install opencv-python
pip install numpy
pip install pytesseract
pip install supervision
pip install ultralytics
```

## Funcionamiento.
### Procesamiento de imagenes

<br> 1. Descargar la base de datos, descomprimir el archivo ZIP y quedarse solo con la carpeta **images**.

> [!IMPORTANT]
> La base de datos de imangenes utilizada para este proyecto pertenece a su resprectivo creador.
> <br><br>Link de la base de datos de las imagenes: https://data.mendeley.com/datasets/nx9xbs4rgx/2

<br> 2. Ejecutar ***change_names.py***.

> [!WARNING]
> Asegurese que la carpeta **images** y el archivo ***change_names.py*** esten en la misma carpeta.

<br> 3. Ejecutar  ***recorta.py***.

> [!IMPORTANT]
> Cambiar el valor de *num_threads* a un valor que este dentro del rango de la cantidad de procesadores lógicos de tu procesador

### Procesamiento de video
> [!IMPORTANT]
> Los videos utilizados para este proyecto pertenecen a sus resprectivos creadores.
> <br><br>Link del video *test* y *test3*: https://www.youtube.com/watch?v=QmwIjn6rwQA
> <br><br>Link del video *test2*: https://www.youtube.com/watch?v=-UQbT7ncCbs&t=55s

<br> 1. Descargar el repositorio o solo el contenido de la carpeta **procesar_videos** y descargar el video del test2, si es que se desea utilizar.

> [!WARNING]
> Asegurese que se hayan descargado correctamente y que esten en la misma carpeta los videos que el archivo ***detector.py***.

<br> 2. Ejecutar el archivo ***detector.py*** que se encuentra en la carpeta **procesar_videos**.

Este proyecto se distribuye bajo la Licencia MIT. Consulta el archivo LICENSE para obtener más detalles.

Si deseas contribuir a este proyecto, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92.

2024 | MrMike92 🐢