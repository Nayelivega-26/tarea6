import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import os

# Ruta relativa al dataset
dataset_path = "dataset"

# Tamaño de las imágenes y tamaño del batch
image_size = (64, 64)
batch_size = 32

# Cargar imágenes desde el directorio
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Crear la red neuronal
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Guardar el modelo entrenado
model.save("modelo_senas.h5")

print("✅ Entrenamiento completado y modelo guardado como modelo_senas.h5")
