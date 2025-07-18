import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

image_dir = "dataset/images"  # folder with SAR grayscale images
mask_dir = "dataset/masks"    # folder with binary masks (oil = 1, background = 0)

def load_data(image_dir, mask_dir, size=(128, 128)):
    X, Y = [], []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            img = img_to_array(load_img(img_path, color_mode="grayscale", target_size=size)) / 255.0
            mask = img_to_array(load_img(mask_path, color_mode="grayscale", target_size=size)) / 255.0
            X.append(img)
            Y.append(mask)
    return np.array(X), np.array(Y)

X, Y = load_data(image_dir, mask_dir)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(128, 128, 1)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    UpSampling2D(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    UpSampling2D(),
    Conv2D(1, (1,1), activation='sigmoid', padding='same')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20, batch_size=8, validation_data=(X_val, Y_val))

model.save("oil_spill_detector.h5")
