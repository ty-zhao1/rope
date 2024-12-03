import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Sequential
import os

# Set up directories
train_dir = '10Knots_split/train'
val_dir = '10Knots_split/val'

# Parameters
batch_size = 32

img_height = 432
img_width = 648

# Define ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,         # Normalize pixel values to [0, 1]
    rotation_range=20,         # Rotate images randomly within ±20 degrees
    width_shift_range=0.2,     # Shift images horizontally by 20%
    height_shift_range=0.2,    # Shift images vertically by 20%
    shear_range=0.2,           # Shear transformations
    zoom_range=0.2,            # Zoom in/out by 20%
    horizontal_flip=True       # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only normalize

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only normalize

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),    # Resize all images to 224x224
    batch_size=32,
    class_mode="categorical"   # One-hot encode labels for categorical crossentropy
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),  # Fewer filters
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),  # Reduce filters here
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),  # Only one larger Conv2D layer
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),  # Still keep the global pooling
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Save the model
# model.save('knot_classifier_model.h5')
model.save('baseline_model.keras')
