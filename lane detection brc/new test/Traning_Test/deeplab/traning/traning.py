import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load DeepLabV3
deeplab_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

# Add segmentation head
x = tf.keras.layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(deeplab_model.output)
x = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=deeplab_model.input, outputs=x)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Load dataset
IMAGE_DIR = "output/deeplab_masks/"
MASK_DIR = "output/deeplab_masks/"

image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

X, Y = [], []

for img_file, mask_file in zip(image_files, mask_files):
    img = cv2.imread(os.path.join(IMAGE_DIR, img_file))
    mask = cv2.imread(os.path.join(MASK_DIR, mask_file), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (128, 128))
    mask = cv2.resize(mask, (128, 128))

    X.append(img / 255.0)
    Y.append(mask / 255.0)

X = np.array(X)
Y = np.array(Y).reshape(-1, 128, 128, 1)

# Train the model
model.fit(X, Y, epochs=10, batch_size=8, validation_split=0.2)

# Save the model
model.save("deeplab_model.h5")
print("🎯 DeepLabV3 Model Training Completed!")
