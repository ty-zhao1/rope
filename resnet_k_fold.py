import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd

# Parameters
img_height = 432
img_width = 648
batch_size = 32
data_dir = '10Knots'

# Data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

# Load all image paths and labels using flow_from_directory
dataset = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Keep the order consistent for splitting
)

# Extract file paths and labels
file_paths = np.array(dataset.filepaths)
labels = np.array(dataset.classes)
num_classes = len(dataset.class_indices)  # Number of classes

# Split the data into train+validation and test sets (e.g., 80% train+validation, 20% test)
trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
    file_paths,
    labels,
    test_size=0.2,  # 20% test set
    stratify=labels,  # Maintain class distribution
    random_state=42
)

print(f"Train+Validation set size: {len(trainval_paths)}")
print(f"Test set size: {len(test_paths)}")

# Define the k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data should only be rescaled
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# K-Fold splitting on the train+validation set
for fold_no, (train_idx, val_idx) in enumerate(kf.split(trainval_paths), start=1):
    print(f"Fold {fold_no}/{k}")
    
    # Get train and validation data for this fold
    fold_train_paths = trainval_paths[train_idx]
    fold_val_paths = trainval_paths[val_idx]
    fold_train_labels = trainval_labels[train_idx]
    fold_val_labels = trainval_labels[val_idx]
    
    print(f"Training set size: {len(fold_train_paths)}, Validation set size: {len(fold_val_paths)}")
    # Create training generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": fold_train_paths, "class": fold_train_labels}),
        directory=None,
        x_col="filename",
        y_col="class",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # Create validation generator
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": fold_val_paths, "class": fold_val_labels}),
        directory=None,
        x_col="filename",
        y_col="class",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    #TODO train models and evaluate them, saving best model
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

    # Compile the model with a low learning rate for initial training
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#TODO load the best model
final_model = None
    
# Create a test generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({"filename": test_paths, "class": test_labels}),
    directory=None,
    x_col="filename",
    y_col="class",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the final model
test_loss, test_accuracy = final_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

