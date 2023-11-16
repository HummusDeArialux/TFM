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
filepath = 'E:/TFM/code/metadata.csv'
metadata = pd.read_csv(filepath)

# Image directory
image_directory = 'E:/TFM/BCN20000/Imagenes_256'

# Define the labels to keep
labels_to_keep = ['melanoma', 'basal cell carcinoma', 'nevus']

# Define the output directory for all labels
output_directory = 'E:/TFM/BCN20000/Imagenes_256_reduced'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the maximum number of "nevus" images to keep
max_nevus_images = 2850

# Shuffle the indices of "nevus" images for random selection
nevus_indices = list(metadata[metadata['diagnosis'] == 'nevus'].index)
random.shuffle(nevus_indices)

# Initialize variables
nevus_count = 0  # Keep track of the number of "nevus" images saved
save_nevus = True  # Flag to control saving of nevus images
rows_to_keep = []  # Create a list to store the indices of rows to keep in the reduced metadata

# Iterate through the metadata and gather image information
for index, row in metadata.iterrows():
    label = row['diagnosis']
    
    if label not in labels_to_keep:
        continue  # Skip images with labels not in the list to keep

    if label == 'nevus' and not save_nevus:
        continue # Skip nevus images if max is reached

    # Form the image file name using the isic_id
    image_filename = row['isic_id'] + '.jpg'
    image_path = os.path.join(image_directory, image_filename)

    # Read and preprocess the image
    image = cv2.imread(image_path)
    
    # Form the output file path
    output_file_path = os.path.join(output_directory, image_filename)

    # Save the image to the output directory
    cv2.imwrite(output_file_path, image)

    if label == 'nevus':
        nevus_count += 1 # If label is nevus add 1 to nevus counter
        if nevus_count >= max_nevus_images:
            save_nevus = False  # If nevus max is reached set save_nevus to False

    # Store the index of this row to keep in the reduced metadata
    rows_to_keep.append(index)

# Create the reduced metadata with only the rows you want to keep
reduced_metadata = metadata.loc[rows_to_keep]

# Save the reduced metadata to 'reducedmetadata.csv'
reduced_metadata.to_csv('E:/TFM/code/reduced_metadata.csv', index=False)