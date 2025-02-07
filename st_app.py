import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

st.title("Gender Detection App")

# Load trained model
model = tf.keras.models.load_model("gendermodel.h5")

# Image Preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image

# **1. Webcam Stream with Stop Button**
st.subheader("Live Camera Feed")
start_camera = st.button("Start Camera")

if start_camera:
    camera = cv2.VideoCapture(0)
    stop_camera = st.button("Stop Camera")
    stframe = st.empty()
    
    while camera.isOpened():
        success, frame = camera.read()
        if not success or stop_camera:
            camera.release()
            break
        
        # Detect Gender
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue  # Skip if face detection fails
            
            processed_face = preprocess_image(face)
            prediction = model.predict(processed_face)[0][0]
            gender = "Male" if prediction < 0.5 else "Female"

            # Debugging: Print Prediction Value
            print(f"Prediction Value: {prediction}, Classified as: {gender}")

            # Draw Box & Label
            color = (255, 0, 0) if gender == "Male" else (255, 20, 147)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        stframe.image(frame, channels="BGR")

# **2. Image Upload Fix**
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]
        gender = "Male" if prediction > 0.5 else "Female"

        # Debugging: Print Prediction Value
        print(f"Prediction Value: {prediction}, Classified as: {gender}")

        st.write(f"Predicted Gender: **{gender}**")
