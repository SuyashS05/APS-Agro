import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths
TRAIN_IMAGE_DIR = r"F:\lane detection brc\laneNet\mask\Train"
VAL_IMAGE_DIR = r"F:\lane detection brc\laneNet\mask\Test"

# Load dataset
def load_dataset(image_dir):
    images = []
    masks = []
    
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize to match model input
            img = img / 255.0  # Normalize

            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale mask
            mask = cv2.resize(mask, (128, 128))
            mask = mask / 255.0  # Normalize mask

            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks).reshape(-1, 128, 128, 1)

# Load train and validation data
X_train, Y_train = load_dataset(TRAIN_IMAGE_DIR)
X_val, Y_val = load_dataset(VAL_IMAGE_DIR)

print(f"Loaded {len(X_train)} training images and {len(X_val)} validation images.")

# Define LaneNet CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary segmentation (lane vs. background)
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=8, validation_data=(X_val, Y_val))

# Save the model
model.save(r"F:\lane detection brc\Traning_Test\lanenet\models\lanenet_model.h5")
print("🎯 LaneNet Model Training Completed!")
