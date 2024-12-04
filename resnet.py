import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

train_dir = '10Knots_split/train'
val_dir = '10Knots_split/val'

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--resnet', type=bool, default=False)
args = parser.parse_args()

img_height = 432
img_width = 648
batch_size = 32
num_classes = 9
lr = args.lr
img_height = 432
img_width = 648

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,         # Normalize pixel values to [0, 1]
    rotation_range=20,         # Rotate images randomly within ±20 degrees
    width_shift_range=0.2,     # Shift images horizontally by 20%
    height_shift_range=0.2,    # Shift images vertically by 20%
    shear_range=0.2,           # Shear transformations
    zoom_range=0.2,            # Zoom in/out by 20%
    horizontal_flip=True       # Randomly flip images horizontally
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode="categorical"
)

if args.resnet:
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
else:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_knot_ResNet50.ckpt', save_best_only=True)

epochs = 20
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)
model_name = 'ResNet50' if args.resnet else 'MobileNetV2'
model.save(f'knot_classifier_{model_name}_{batch_size}_{lr}_{epochs}.keras')