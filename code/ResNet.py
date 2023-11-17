import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import to_categorical
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# LOAD AND PREPROCESS DATA
# Load file
filepath = 'E:/TFM/code/reduced_metadata.csv'
metadata = pd.read_csv(filepath)  # Use the specified file path

# Delete non-informative columns
col_drop = ['attribution', 'copyright_license', 'diagnosis_confirm_type', 'image_type', 'lesion_id']
data = metadata.drop(columns=col_drop)  # Elimination of the specified columns

# Define a list to store image information
image_info = []

# Image directory
image_directory = 'E:/TFM/BCN20000/Imagenes_256_reduced'

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

# CREATE RESNET MODEL
# Load the pre-trained ResNet50 model with weights from ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))

# Freeze layers up to block4
for layer in base_model.layers:
    if 'conv4_block' in layer.name:
        break
    layer.trainable = False
    
# Define hyperparameters
num_epochs = 30 # Number of epochs
batch_size = 128 # Batch size
learning_rate = 0.001 # Learning rate
dropout_rate = 0.25 # Dropout rate

# Add custom layers on top of the pre-trained model
x = base_model.output # Base model
x = GlobalAveragePooling2D()(x) # Global Average Pooling to reduce spatial dimensions
x = BatchNormalization()(x) # Batch Normalization for improved training stability
x = Dense(256, activation='relu')(x) # Dense layer with ReLU activation
x = Dropout(dropout_rate)(x) # Dropout for regularization
predictions = Dense(num_classes, activation='softmax')(x) # Final Dense layer for classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',  # Because it's a multi-class classification problem
              metrics=['accuracy']) # Evaluate model based on accuracy

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Fit the model
history = model.fit(
    X_train, # Training images
    y_train, # One-hot encoded training labels
    validation_data=(X_val, y_val), # Validation data with one-hot encoded labels
    epochs=num_epochs, # Number of training epochs
    batch_size=batch_size, # Batch size
    callbacks=[lr_scheduler] # Add the learning rate scheduler
)

# Save the model to a specific path
model.save('E:/TFM/Results with code/Trained_models/ResNet50.h5')

# EVALUATE MODEL
# Plot training and validation loss
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# Plot training and validation accuracy
plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Predictions on test set
y_pred = model.predict(X_test)

# Convert one-hot encoded predictions back to class labels
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
y_true_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))

# Calculate and print various evaluation metrics
# Calculate accuracy
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f'Test accuracy: {accuracy:.2f}')

# Calculate precision
precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
print(f'Precision: {precision:.2f}')

# Calculate sensitivity (recall)
sensitivity = recall_score(y_true_labels, y_pred_labels, average='weighted')
print(f'Sensitivity (Recall): {sensitivity:.2f}')

# Calculate F1-Score
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
print(f'F1-Score: {f1:.2f}')

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
print(f"Cohen's Kappa: {kappa:.2f}")

# Generate a classification report
report = classification_report(y_true_labels, y_pred_labels)
print("Classification Report:")
print(report)

# Calculate the confusion matrix
confusion = confusion_matrix(y_true_labels, y_pred_labels)
print("Confusion Matrix:")
print(confusion)

# Get class labels
classes = unique_labels(y_true_labels, y_pred_labels)

# Plot a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
