# predictor.py
# Contains the necessary code for making predictions
import os
import tensorflow as tf
import numpy as np
import os

# Import and define model as model
model = tf.keras.models.load_model(os.getcwd() + '/DenseNet121.h5')

# Define predict function
def predict_image(def_image):
    # Make the prediction
    result = model.predict(def_image)

    # Create dictionary with pertinent label encoding
    class_names = {0: 'Basal cell carcinoma',
                   1: 'Melanoma',
                   2: 'Nevus'}

    # Save the results as variables
    prediction = int(np.argmax(result))
    alteration = class_names[prediction]
    result_probability = round(result[0][prediction], 2)

    return alteration, result_probability, result, prediction
