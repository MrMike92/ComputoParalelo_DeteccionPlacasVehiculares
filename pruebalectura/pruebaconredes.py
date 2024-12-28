import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Configuración
IMG_SIZE = (28, 28)  # Tamaño de las imágenes (asumimos 28x28)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_CSV_PATH = 'PROYECTO_REDES/digital_letters.csv'  # Ruta del archivo CSV de entrenamiento
TEST_IMAGES_PATH = 'PROYECTO_REDES/imagess/'  # Ruta de las imágenes de prueba

# Cargar y preprocesar los datos desde CSV
def load_data_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    images = data.iloc[:, 1:-1].values / 255.0  # Normalizar los valores de píxeles
    labels = data.iloc[:, -1].values
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    labels_idx = np.array([label_to_idx[label] for label in labels])
    return images, labels_idx, label_to_idx

images, labels, label_to_idx = load_data_from_csv(TRAIN_CSV_PATH)
class_names = {v: k for k, v in label_to_idx.items()}

# Dividir el conjunto de datos
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Cargar y preprocesar las imágenes de prueba
def load_test_images(test_path, img_size):
    test_images = []
    image_files = []

    for img_file in os.listdir(test_path):
        img_path = os.path.join(test_path, img_file)
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            test_images.append(img_array.flatten())
            image_files.append(img_file)
        except Exception as e:
            print(f"Error cargando la imagen {img_path}: {e}")

    return np.array(test_images), image_files

test_images, test_image_files = load_test_images(TEST_IMAGES_PATH, IMG_SIZE)

# Construcción de la DBN manual con autoencoders
class DBNModel(tf.keras.Model):
    def __init__(self, hidden_layers):
        super(DBNModel, self).__init__()
        self.encoders = []
        self.decoders = []
        for units in hidden_layers:
            self.encoders.append(layers.Dense(units, activation='relu'))
            self.decoders.append(layers.Dense(units, activation='relu'))

        self.classifier = models.Sequential([
            layers.Dense(hidden_layers[-1], activation='relu'),
            layers.Dense(len(class_names), activation='softmax')
        ])

    def call(self, x, training=False):
        for encoder in self.encoders:
            x = encoder(x)
        if training:
            for decoder in self.decoders[::-1]:
                x = decoder(x)
        return self.classifier(x)

# Definir modelo y entrenar por etapas
hidden_layers = [256, 128, 64]
model = DBNModel(hidden_layers)

# Compilación y entrenamiento del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Entrenando la DBN manual...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluación del modelo en el conjunto de validación
print("Evaluando la DBN en el conjunto de validación...")
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Precisión en el conjunto de validación: {val_accuracy:.2f}")

# Predicción sobre imágenes de prueba
def decode_predictions(predictions, class_names):
    return [class_names[np.argmax(pred)] for pred in predictions]

def predict_plate_letters(images, model, class_names):
    predictions = model.predict(images)
    letters = decode_predictions(predictions, class_names)
    return ''.join(letters)

print("Realizando predicciones en las imágenes de prueba...")
all_predictions = []
for test_image, file_name in zip(test_images, test_image_files):
    letters = predict_plate_letters(np.expand_dims(test_image, axis=0), model, class_names)
    all_predictions.append((file_name, letters))

# Mostrar resultados de las predicciones
print("Resultados de las imágenes de prueba:")
for file, pred in all_predictions:
    print(f"Imagen: {file}, Predicción de la placa: {pred}")
