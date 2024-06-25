import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import os

def load_nested_dir(data_dir):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    num_classes = len(class_names)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:  # Ensure the image is loaded correctly
                    image = cv2.resize(image, (28, 28))  # Resize to a fixed size (28x28)
                    image = image / 255.0  # Normalize pixel values to [0, 1]
                    images.append(image)
                    labels.append(int(class_name))

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Add channel dimension
    labels = np.array(labels)
    labels = to_categorical(labels, 127)

    return images, labels

###### Usage ############
# data_dir = '../data/handwritting_characters_database/curated'
# images, labels, class_names = load_nested_dir(data_dir)

def labels_to_name_nested_dir(data_dir):
    # Get sorted directory names, assuming they are named as ASCII values of the characters
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    # Convert directory names from ASCII values to characters
    class_names = [chr(int(d)) for d in class_dirs]
    return class_names