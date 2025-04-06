import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Preprocessing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Contrast Enhancement: Histogram Equalization
def histogram_equalization(image):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Contrast Enhancement: CLAHE
def apply_clahe(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose preprocessing operation", 
                          ["Smoothening", "Shearing", "Sharpening", "Contrast: Histogram Equalization", "Contrast: CLAHE"])

    if option == "Smoothening":
        kernel_size = st.slider("Kernel Size (odd numbers only)", 3, 21, 5, step=2)
        result = smoothen_image(image, kernel_size)

    elif option == "Shearing":
        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.2)
        result = shear_image(image, shear_factor)

    elif option == "Sharpening":
        result = sharpen_image(image)

    elif option == "Contrast: Histogram Equalization":
        result = histogram_equalization(image)

    elif option == "Contrast: CLAHE":
        result = apply_clahe(image)

    st.image(result, caption=f"{option} Applied", use_column_width=True)
