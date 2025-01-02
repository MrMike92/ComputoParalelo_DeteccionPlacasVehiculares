import cv2

def main():
    # Cargar el clasificador preentrenado para vehículos
    car_cascade = cv2.CascadeClassifier('PROYECTO/haarcascade_car.xml')

    if car_cascade.empty():
        print("Error: No se pudo cargar el clasificador de vehículos.")
        exit()

    cap = cv2.VideoCapture('PROYECTO/video.mp4')

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error: No se puede abrir el archivo de video")
        exit()

    frame_number = 0  # Inicializar el conteo de frame

    # Permite sacar informacion del video analizado para usarlo en el de salida
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Configuracion de formato del video de salida
    output_video_path = 'PROYECTO_REDES/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificador para el archivo de salida
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    tracker = None
    bbox = None

    while cap.isOpened():
        ret, frame = cap.read()  # Lee el video frame a frame
        if not ret:
            break

        frame_number += 1  # Suma 1 al conteo de frame
        print("Numero de frame: ", frame_number)

        if tracker is None:  # Si no hay un rastreador activo
            # Convertir el frame a escala de grises para la detección
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detectar vehículos en el frame
            vehicles = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            if len(vehicles) > 0:
                # Seleccionar el primer vehículo detectado y crear un rastreador
                x, y, w, h = vehicles[0]

                # Expandir los límites para abarcar todo el vehículo
                expansion_factor = 0.2  # Aumentar un 20% alrededor del cuadro original
                x = max(0, int(x - w * expansion_factor))
                y = max(0, int(y - h * expansion_factor))
                w = min(frame_width - x, int(w * (1 + 2 * expansion_factor)))
                h = min(frame_height - y, int(h * (1 + 2 * expansion_factor)))

                bbox = (x, y, w, h)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
        else:
            # Actualizar la posición del rastreador
            success, bbox = tracker.update(frame)

            if not success:  # Si el rastreador pierde el objeto
                tracker = None
                bbox = None

        # Dibujar el cuadro de seguimiento si hay uno activo
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Escribir el frame procesado en el video de salida
        out.write(frame)

        # Mostrar el frame con las detecciones y el rastreo
        cv2.imshow('Detección y Rastreo de Vehículos', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
