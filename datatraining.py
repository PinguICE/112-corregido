import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Aumento de datos aleatorio (Cambio de tamaño, rotación, giros, zoom, transformaciones) usando ImageDataGenerator 
training_data_generator = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# Directorio de la imagen de entrenamiento
training_image_directory = "training_dataset"

# Generación de aumento de datos procesados para imágenes de entrenamiento
training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(180,180))

# Aumento de datos aleatorio (cambio de tamaño) usando ImageDataGenerator
validation_data_generator = ImageDataGenerator(rescale = 1.0/255)

# Directorio de la imagen de validación
validation_image_directory = "validation_dataset"

# Generación de aumento de datos procesados para imágenes de validación
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(180,180))

model = tf.keras.models.Sequential([
    # 1a Capa de convolución y capa pooling 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)), 
    tf.keras.layers.MaxPooling2D(2, 2),
    # 2a Capa de convolución y capa pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # 3a Capa de convolución y capa pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # 4a Capa de convolución y capa pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Aplanar los resultados para ingresarlos a la capa densa
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dropout(0.5), 
    # Capa de clasificación 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(2, activation='softmax') 
])

# python3 datatraining.py