# Detección de placas vehiculares.
Un detector de placas vehiculares la cual usa paralelismo para la reducción de tiempo.

## Instrucciones de uso.

- Clona este repositorio.
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python:
    - Las versiones que soporta OpenCV son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
    - Las versiones que soporta numpy son 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
- Instalar las siguentes bibliotecas:
```python
# Copiar y pegar lo siguente en el CMD de Windows.
pip install opencv-python
pip install numpy
```

## Funcionamiento.
Por el momento solo funciona los siguentes códigos que se encuentran en la carpeta "procesar_imagenes"

<br> 1. Descargar la base de datos, descomprimir el archivo ZIP y quedarse solo con la carpeta "images".

> [!IMPORTANT]
> La base de datos de imangenes utilizada para este proyecto pertenece a su resprectivo creador.
> <br><br>Link de la base de datos de las imagenes: https://data.mendeley.com/datasets/nx9xbs4rgx/2

<br> 2. Ejecutar "change_names.py".

> [!WARNING]
> Asegurese que la carpeta "images" y el archivo "change_names.py" esten en la misma carpeta.

<br> 3. Ejecutar "recorta.py".

> [!IMPORTANT]
> Cambiar el valor de num_threads a un valor que este dentro del rango de la cantidad de procesadores lógicos de tu procesador

2024 | MrMike92 🐢
