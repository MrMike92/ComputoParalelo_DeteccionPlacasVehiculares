# Conectividad-8

import cv2
import numpy as np

class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
        self.colors = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8)
    
    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        # Comprimir la ruta
        while x != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def detect_plates(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un umbral para resaltar las áreas de interés
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar los contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar los contornos para identificar las posibles placas
    plates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 2 < aspect_ratio < 10 and 10000 < area < 90000:
            plates.append((x, y, w, h))
    
    # Recortar las regiones correspondientes a las placas
    plate_images = [image[y:y+h, x:x+w] for (x, y, w, h) in plates]
    
    # Etiquetar los componentes conectados en cada placa
    for plate_image in plate_images:
        # Convertir la imagen a escala de grises
        plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # Aplicar un umbral para obtener una imagen binaria
        _, plate_thresh = cv2.threshold(plate_gray, 75, 255, cv2.THRESH_BINARY)
        # Encontrar los contornos en la imagen binaria
        contours, _ = cv2.findContours(plate_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Crear una instancia de UnionFind
        uf = UnionFind(len(contours))
        # Unir los componentes conectados
        for i, contour in enumerate(contours):
            x, y, _, _ = cv2.boundingRect(contour)
            uf.union(i, i)
            for j, other_contour in enumerate(contours[i + 1:], start=i + 1):
                x_other, y_other, _, _ = cv2.boundingRect(other_contour)
                if (abs(x - x_other) < 20 and abs(y - y_other) < 20) or (abs(x - x_other) < 20 and abs(y - y_other) == 0) or (abs(x - x_other) == 0 and abs(y - y_other) < 20):
                    uf.union(i, j)
        # Etiquetar los componentes conectados
        for i, contour in enumerate(contours):
            color = uf.colors[uf.find(i)].tolist()
            cv2.drawContours(plate_image, [contour], -1, color, -1)
        
        # Segunda pasada
        for i, contour in enumerate(contours):
            color = uf.colors[uf.find(i)].tolist()
            cv2.drawContours(plate_image, [contour], -1, color, 2)
    
    return plate_images

# Procesar 50 imágenes
for i in range(1, 316):
    image_path = f'dataset/croppet/cropped_parking_lot_{i}.jpg'
    car_image = cv2.imread(image_path)

    # Detectar las placas en la imagen del automóvil
    plate_images = detect_plates(car_image)

    # Mostrar las placas detectadas
    for j, plate_image in enumerate(plate_images):
        cv2.imshow(f"Conectividad-8_#{i}", plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()