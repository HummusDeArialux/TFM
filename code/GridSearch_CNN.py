# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:42:53 2023

@author: Arialux
"""
# Import libraries
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import numpy as np
from tensorflow.keras.utils import to_categorical
import gc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# LOAD AND PREPROCESS DATA
# Mount Google Drive
drive.mount('/content/drive')

# Define file paths in Google Drive
metadata_file_path = '/content/drive/My Drive/Colab_Notebooks/test_metadata.csv'
image_directory = '/content/drive/My Drive/Colab_Notebooks/Imagenes_test'

# Load the metadata file
metadata = pd.read_csv(metadata_file_path)

# Delete non-informative columns
col_drop = ['attribution', 'copyright_license', 'diagnosis_confirm_type', 'image_type', 'lesion_id']
data = metadata.drop(columns = col_drop) # Eliminación de las columnas

# Define a list to store image information
image_info = []

# Iterate through the metadata and gather image information
for index, row in metadata.iterrows():
    # Form the image file name using the isic_id
    image_filename = row['isic_id'] + '.jpg'
    image_path = os.path.join(image_directory, image_filename)
    label = row['diagnosis']

    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = image / 255.0  # Normalize the pixel values to the range [0, 1]

    # Store the image and label information
    image_info.append((image, label))

# Extract images and labels from image_info
images = np.array([item[0] for item in image_info])  # Convert image data to NumPy arrays
labels = [item[1] for item in image_info]

# Unique number of diagnosis
num_classes = len(Counter(labels))

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Convert encoded labels to one-hot encoding
y_one_hot = to_categorical(y_encoded, num_classes=num_classes)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, y_one_hot, test_size=0.2, random_state=2023)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2023)

# CLEAN RAM
del images
del image
del labels
del data
del image_info
del metadata
del X_temp
del y_temp
del y_one_hot
gc.collect()

# GRID SEARCH
# Define hyperparameters
learning_rates = [0.1, 0.01, 0.001, 0.0001]
dropout_rates = [0.10, 0.15, 0.25, 0.35]
num_epochs = [15, 20, 25, 30]

results = {}  # Dictionary to store results for different hyperparameters on val accuracy
results2 = {}  # Dictionary to store results for different hyperparameters on test accuracy

# Function to create and train the model
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, dropout_rate, epochs):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(dropout_rate))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(dropout_rate))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    return history.history['val_accuracy'][-1], test_accuracy

# Perform grid search
for learning_rate in learning_rates:
    for dropout_rate in dropout_rates:
        for epochs in num_epochs:
            val_accuracy, test_accuracy = train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                                                      learning_rate, dropout_rate, epochs)
            results[(learning_rate, dropout_rate, epochs)] = val_accuracy
            results2[(learning_rate, dropout_rate, epochs)] = test_accuracy

# Find the best hyperparameters for validation accuracy
best_hyperparameters = max(results, key=results.get)
print(f"Best hyperparameters for validation accuracy: {best_hyperparameters} - Validation Accuracy: {results[best_hyperparameters]}")

# Find the best hyperparameters for test accuracy
best_hyperparameters_test = max(results2, key=results2.get)
print(f"Best hyperparameters for test accuracy: {best_hyperparameters_test} - Test Accuracy: {results2[best_hyperparameters_test]}")

# Plot the grid search results for validation accuracy
hyperparameters_val = list(results.keys())
accuracies_val = list(results.values())
learning_rates_val, dropout_rates_val, epochs_val = zip(*hyperparameters_val)

log_learning_rates_val = np.log10(learning_rates_val)

fig_val = plt.figure(figsize=(10, 8))
ax_val = fig_val.add_subplot(111, projection='3d')

sc_val = ax_val.scatter(log_learning_rates_val, dropout_rates_val, epochs_val, c=accuracies_val, cmap='YlGnBu', marker='o')

ax_val.set_xlabel('Log Learning Rate (log10 scale)')
ax_val.set_ylabel('Dropout Rate')
ax_val.set_zlabel('Number of Epochs')
cbar_val = fig_val.colorbar(sc_val)
cbar_val.set_label('Validation Accuracy')

plt.title("Grid Search for Hyperparameters (Validation Accuracy)")
plt.show()

# Plot the grid search results for test accuracy
hyperparameters_test = list(results2.keys())
accuracies_test = list(results2.values())
learning_rates_test, dropout_rates_test, epochs_test = zip(*hyperparameters_test)

log_learning_rates_test = np.log10(learning_rates_test)

fig_test = plt.figure(figsize=(10, 8))
ax_test = fig_test.add_subplot(111, projection='3d')

sc_test = ax_test.scatter(log_learning_rates_test, dropout_rates_test, epochs_test, c=accuracies_test, cmap='YlGnBu', marker='o')

ax_test.set_xlabel('Log Learning Rate (log10 scale)')
ax_test.set_ylabel('Dropout Rate')
ax_test.set_zlabel('Number of Epochs')
cbar_test = fig_test.colorbar(sc_test)
cbar_test.set_label('Test Accuracy')

plt.title("Grid Search for Hyperparameters (Test Accuracy)")
plt.show()

