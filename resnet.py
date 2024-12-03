import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_dir = '10Knots_split/train'
val_dir = '10Knots_split/val'

# Parameters
img_height = 432
img_width = 648
batch_size = 32
num_classes = 9  # Number of knot classes
lr = 1e-3

# Data augmentation and loading
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
# )

# val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

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

# Load the ResNet50 model pre-trained on ImageNet, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# Freeze the base model
base_model.trainable = False

# Add custom layers on top of ResNet
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with a low learning rate for initial training
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_knot_ResNet50.ckpt', save_best_only=True)

# Train the model (initial training of custom top layers only)
epochs = 20
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save(f'knot_classifier_ResNet50_{batch_size}_{lr}_{epochs}.keras')