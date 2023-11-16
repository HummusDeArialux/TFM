# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:03:15 2023
@author: Arialux
"""
# Import libraries
import os
import cv2
import pandas as pd
import random

# Set seed
random_seed = 2023
random.seed(random_seed)

# Load file
filepath = 'E:/TFM/code/reduced_metadata.csv'
metadata = pd.read_csv(filepath)

# Shuffle the entire dataset randomly
metadata = metadata.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Image directory
image_directory = 'E:/TFM/BCN20000/Imagenes_256_reduced'

# Define the labels you want to keep
labels_to_keep = ['melanoma', 'basal cell carcinoma', 'nevus']

# Define the output directory for all labels
output_directory = 'E:/TFM/BCN20000/Imagenes_test'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the maximum number of images per class
total_images_to_keep = 1000

# Create a dictionary to keep track of the count for each class
class_counts = {label: 0 for label in labels_to_keep}

# Create a list to store the indices of rows to keep in the reduced metadata
rows_to_keep = []

# Iterate through the metadata and gather image information
for _, row in metadata.iterrows():
    label = row['diagnosis']
    
    # If the maximum number of images for a class is reached, skip the image
    if class_counts[label] >= total_images_to_keep / len(labels_to_keep):
        continue

    # Form the image file name using the isic_id
    image_filename = row['isic_id'] + '.jpg'
    image_path = os.path.join(image_directory, image_filename)

    # Read and preprocess the image
    image = cv2.imread(image_path)
    
    # Form the output file path
    output_file_path = os.path.join(output_directory, image_filename)

    # Save the image to the output directory
    cv2.imwrite(output_file_path, image)

    # Increment the count for the current class
    class_counts[label] += 1

    # Store the index of this row to keep in the reduced metadata
    rows_to_keep.append(_)

    # Check if the required number of images for each class is reached
    if all(count >= total_images_to_keep for count in class_counts.values()):
        break  # Stop once the required number of images for all classes is reached

# Create the reduced metadata with only the rows you want to keep
test_metadata = metadata.loc[rows_to_keep]

# Save the reduced metadata to 'reducedmetadata.csv'
test_metadata.to_csv('E:/TFM/code/test_metadata.csv', index=False)

print('Subset of 1000 images created with balanced classes. Reduced metadata saved.')

