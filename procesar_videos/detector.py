import cv2

def main():
    cap = cv2.VideoCapture('videos/video.mp4')

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
       print("Error: No se puede abrir el archivo de video")
       exit()

    frame_number = 0 # Inicializar el conteo de frame

    #Permite sacar informacion del video analizado para usarlo en el de salida
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Configuracion de formato del video de salida
    output_video_path = 'videos/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificador para el archivo de salida
    out = cv2.VideoWriter('videos/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret,frame = cap.read() # Lee el video frame a frame
        frame_number +=1  # Suma 1 al conteo de frame
        
        if not ret:
           break 

        print("Numero de frame: ",frame_number)
        out.write(frame) #Guarda el frame para construir el video

        # Mostrar el frame mientra se guardan para el video
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
   
if __name__ == "__main__":
    main()