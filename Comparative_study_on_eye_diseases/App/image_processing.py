import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import streamlit as st


def lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def clahecc(image):

    # Check if the image is not already grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(grayscale_image)

    # Contrast Correction (optional)
    clahe_cc_image = np.clip(2.5 * clahe_image - 128, 0, 255).astype(np.uint8)
    return clahe_cc_image


def gabor(image):
    # Define parameters for Gabor filter
    ksize = 3  # Kernel size
    sigma = 10
    # Standard deviation of the Gaussian envelope
    theta = np.pi/4  # Orientation of the normal to the parallel stripes of the Gabor function
    lambda_ = 4  # Wavelength of the sinusoidal factor
    gamma = 0.25  # Spatial aspect ratio
    phi = 1  # Phase offset of the sinusoidal factor

    # Generate Gabor filter
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambda_, gamma, phi, ktype=cv2.CV_32F)

    # Apply Gabor filter to the image
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered_image


def get_image(image, num=4):

    final_size = (224, 224)
    if num == 1:
        # Convert the grayscale image to a three-channel grayscale image
        img = cv2.resize(image, final_size)
        return np.array(img)

    elif num == 2:
        a = lab(image)
        img = hsv(a)
        img = cv2.resize(img, final_size)
        return np.array(img)

    elif num == 3:
        a = lab(image)
        img = clahecc(a)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, final_size)
        return np.array(img)

    else:
        a = hsv(image)
        img = gabor(a)
        img = cv2.resize(img, final_size)
        return np.array(img)
