# utils.py
import os
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
