import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Preprocessing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def scale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def contrast_stretching(image):
    in_min = np.min(image)
    in_max = np.max(image)
    out = (image - in_min) * (255 / (in_max - in_min))
    return out.astype(np.uint8)

def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (int(cols + shear_factor * rows), rows))

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose preprocessing operation", 
                          ["Scale", "Contrast Stretching", "Shearing", "Sharpening"])

    if option == "Scale":
        scale_percent = st.slider("Scale (%)", 10, 200, 100)
        result = scale_image(image, scale_percent)

    elif option == "Contrast Stretching":
        result = contrast_stretching(image)

    elif option == "Shearing":
        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.2)
        result = shear_image(image, shear_factor)

    elif option == "Sharpening":
        result = sharpen_image(image)

    st.image(result, caption=f"{option} Applied", use_column_width=True)
