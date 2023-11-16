# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:31:40 2023

@author: Arialux
"""
# Import libraries
import os
import cv2
import pandas as pd

# Load file
filepath = 'E:\TFM\code\reduced_metadata.csv'
dataset = pd.read_csv(filepath)

# Examine file
print(f'Variables: {dataset.columns}')
print(f'Number of images: {len(dataset)}')

# Delete non-informative columns
col_drop = ['attribution', 'copyright_license', 'diagnosis_confirm_type', 'image_type', 'lesion_id']
data = dataset.drop(columns = col_drop)
print(f'Variables: {data.columns}')

# Data exploration
diagnosis_counts = data['diagnosis'].value_counts()

# Creation of diagnosis table
diagnosis_table = pd.DataFrame({'Diagnosis': diagnosis_counts.index, 'Sample Size': diagnosis_counts.values})
diagnosis_table['Percentage'] = (diagnosis_table['Sample Size'] / diagnosis_table['Sample Size'].sum()) * 100
diagnosis_table['Percentage'] = diagnosis_table['Percentage'].round(2)

# Display the resulting table
print(diagnosis_table)

# Export the DataFrame to a CSV file
diagnosis_table.to_csv('diagnosis_table.csv', index=False)

# Other info
print(data.sex.value_counts())
print(data.anatom_site_general.value_counts())
print(data.melanocytic.value_counts())

# Image directory
image_directory = 'E:\TFM\BCN20000\Imagenes_256_reduced'

# Function to gather image information recursively
def gather_image_info(directory):
    """
Recursively traverses the specified directory to gather information about
images with the '.JPG' extension. Retrieves the height, width, and channels
of each image using OpenCV.

Parameters:
- directory (str): The path to the directory containing images.

Returns:
- list: A list of dictionaries, where each dictionary contains image information
        including height, width, and channels.
    """
    image_info = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.JPG'):  # Images are in JPG format
                file_path = os.path.join(root, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    height, width, channels = image.shape
                    image_info.append({
                        'Height': height,
                        'Width': width,
                        'Channels': channels
                    })
    return image_info

# Call the function to gather image information
image_info = gather_image_info(image_directory)

# Create a DataFrame to store the gathered information
image_df = pd.DataFrame(image_info)

# Print necessary info
print(image_df.Height.value_counts())
print(image_df.Width.value_counts())
print(image_df.Channels.value_counts())