import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define UNet model
def unet_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # Encoder (Downsampling)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)

    # Decoder (Upsampling)
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, 3, activation="relu", padding="same")(up5)
    conv5 = Conv2D(256, 3, activation="relu", padding="same")(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, 3, activation="relu", padding="same")(up6)
    conv6 = Conv2D(128, 3, activation="relu", padding="same")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate()([up7, conv1])
    conv7 = Conv2D(64, 3, activation="relu", padding="same")(up7)
    conv7 = Conv2D(64, 3, activation="relu", padding="same")(conv7)

    # Output layer (Binary mask)
    outputs = Conv2D(1, 1, activation="sigmoid")(conv7)

    model = Model(inputs, outputs)
    return model

# Compile the UNet model
model = unet_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Dataset paths
IMAGE_DIR = "output/deeplab_masks/"
MASK_DIR = "output/deeplab_masks/"

# Load dataset
image_files = sorted(os.listdir(IMAGE_DIR))
mask_files = sorted(os.listdir(MASK_DIR))

X, Y = [], []

for img_file, mask_file in zip(image_files, mask_files):
    img = cv2.imread(os.path.join(IMAGE_DIR, img_file))
    mask = cv2.imread(os.path.join(MASK_DIR, mask_file), cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (128, 128))
    mask = cv2.resize(mask, (128, 128))

    X.append(img / 255.0)  # Normalize image
    Y.append(mask / 255.0)  # Normalize mask

X = np.array(X)
Y = np.array(Y).reshape(-1, 128, 128, 1)

# Train the model
history = model.fit(X, Y, epochs=20, batch_size=8, validation_split=0.2)

# Save the trained model
model.save("unet_lane_segmentation.h5")
print("🎯 UNet Model Training Completed!")

# Plot training history
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("UNet Training Performance")
plt.legend()
plt.show()
