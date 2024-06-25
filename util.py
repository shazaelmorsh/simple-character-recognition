import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def one_hot_to_ascii(one_hot, offset = 0):
    # Find the index of the 1 in the one-hot encoded array
    index = np.argmax(one_hot)
    # Calculate the corresponding ASCII value
    ascii_value = index + offset
    # Convert the ASCII value to its character representation
    ascii_char = chr(ascii_value)
    return ascii_char
