import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Preprocessing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# 1. Smoothening
def smoothen_image(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 2. Shearing
def shear_image(image, shear_factor):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, M, (int(cols + shear_factor * rows), rows))

# 3. Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# 4. Negative
def negative_image(image):
    return 255 - image

# 5. Grayscale
def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 6. Edge Detection (Canny)
def edge_detection(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# 7. Histogram Equalization
def histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

# 8. Brightness Increase
def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return bright

# 9. Rotation
def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

# 10. Flip (Horizontal)
def flip_image(image):
    return cv2.flip(image, 1)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose preprocessing operation", 
                          ["Smoothening", "Shearing", "Sharpening", "Negative", 
                           "Grayscale", "Edge Detection", "Histogram Equalization", 
                           "Brightness Increase", "Rotation", "Flip Horizontal"])

    if option == "Smoothening":
        ksize = st.slider("Kernel Size (odd)", 3, 21, 5, step=2)
        result = smoothen_image(image, ksize)

    elif option == "Shearing":
        shear_factor = st.slider("Shear Factor", 0.0, 1.0, 0.2)
        result = shear_image(image, shear_factor)

    elif option == "Sharpening":
        result = sharpen_image(image)

    elif option == "Negative":
        result = negative_image(image)

    elif option == "Grayscale":
        result = grayscale_image(image)
        st.image(result, caption=f"{option} Applied", use_column_width=True, channels="GRAY")
        st.stop()

    elif option == "Edge Detection":
        t1 = st.slider("Threshold 1", 0, 300, 100)
        t2 = st.slider("Threshold 2", 0, 300, 200)
        result = edge_detection(image, t1, t2)

    elif option == "Histogram Equalization":
        result = histogram_equalization(image)

    elif option == "Brightness Increase":
        brightness = st.slider("Brightness Level", 0, 100, 30)
        result = increase_brightness(image, brightness)

    elif option == "Rotation":
        angle = st.slider("Rotation Angle", -180, 180, 45)
        result = rotate_image(image, angle)

    elif option == "Flip Horizontal":
        result = flip_image(image)

    st.image(result, caption=f"{option} Applied", use_column_width=True)
