import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

test_dir = '10Knots_split/test'

img_height = 432
img_width = 648
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)

models_dict = {
    'baseline_simple': models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax'),
    ]),
    'baseline': models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'), 
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(9, activation='softmax')
    ]),
    'Resnet50': models.Sequential([
        ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ]),
    'MobileNetV2': models.Sequential([
        MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ]),
}

for file in os.listdir('.'):
    if file.endswith('.keras'):
        if 'baseline_model_simple' in file:
            model = models_dict['baseline_simple']
        elif 'baseline_model' in file:
            model = models_dict['baseline']
        elif 'ResNet50' in file:
            model = models_dict['Resnet50']
        elif 'MobileNetV2' in file:
            model = models_dict['MobileNetV2']
        
        model.load_weights(file)
        
        print(f'Evaluating model {file}')
        
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        model.evaluate(test_generator)
