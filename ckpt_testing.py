import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import ResNet50, MobileNetV2

img_height = 432
img_width = 648

num_classes = 9
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
# Freeze the base model
base_model.trainable = False

# Add custom layers on top of ResNet
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

checkpoint_path = "best_knot_MobileNetV2.ckpt"  # Replace with the actual path
checkpoint = tf.train.Checkpoint(model=model)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

