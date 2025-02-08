import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained gender classification model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gendermodel.keras")

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image

# Title
st.title("Live Gender Detection App")

# **1. Live Camera Input (Works on Mobile & PC)**
st.subheader("Live Camera Feed")
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, caption="Captured Image", use_column_width=True)
        
        # Process and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        gender = "Male" if prediction < 0.5 else "Female"

        st.write(f"Predicted Gender: **{gender}**")

# **2. Image Upload Option**
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        gender = "Male" if prediction < 0.5 else "Female"

        st.write(f"Predicted Gender: **{gender}**")
