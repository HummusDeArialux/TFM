# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:35:26 2023

@author: Arialux
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Pre-process image
image = cv2.imread('E:\TFM\prediction_images\image3.jpg')
resized_image = cv2.resize(image, (256, 256))
normalized_image = resized_image / 255.0

# Expand dimensions to match the model's expected input shape
final_image = np.expand_dims(normalized_image, axis=0)

# Load model
model = tf.keras.models.load_model('E:\TFM\Results with code\Trained_models\DenseNet121.h5')

# Create a dictionary mapping class names to numerical labels
class_names = {0:'basal cell carcinoma',
               1:'melanoma',
               2:'nevus'}

# Make prediction and get results
result = model.predict(final_image)
prediction = int(np.argmax(result))
alteration = class_names[prediction]
probability_values = result[0]
result_probability = round(result[0][prediction], 2)

# Plot the bar chart
plt.bar(class_names.values(), probability_values, color=['blue', 'orange', 'green'])
plt.title('Probability Distribution')
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.show()

print("Predicted class:", prediction)
print("The alteration shown is:", alteration)
print("Probability of the predicted class:", result_probability)