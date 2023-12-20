# predictor.py
# Contains the necessary code for making predictions
import os
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model(os.getcwd() + '/DenseNet121.h5')


def predict_image(def_image):
    result = model.predict(def_image)

    class_names = {0: 'Basal cell carcinoma',
                   1: 'Melanoma',
                   2: 'Nevus'}

    prediction = int(np.argmax(result))
    alteration = class_names[prediction]
    result_probability = round(result[0][prediction], 2)

    return alteration, result_probability, result, prediction
