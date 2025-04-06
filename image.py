import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Preprocessing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Smoothening
def smoothen_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Shearing
def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (int(cols + shear_factor * rows), rows))

# Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Negative transformation
def negative_image(image):
    return 255 - image

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose preprocessing operation", 
                          ["Smoothening", "Shearing", "Sharpening", "Negative"])

    if option == "Smoothening":
        kernel_size = st.slider("Kernel Size (odd numbers only)", 3, 21, 5, step=2)
        result = smoothen_image(image, kernel_size)

    elif option == "Shearing":
        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.2)
        result = shear_image(image, shear_factor)

    elif option == "Sharpening":
        result = sharpen_image(image)

    elif option == "Negative":
        result = negative_image(image)

    st.image(result, caption=f"{option} Applied", use_column_width=True)
