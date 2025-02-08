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

# **1. Live Video Feed (PC & Mobile Compatible)**
st.subheader("Live Camera Feed")

# Start button
start_camera = st.checkbox("Enable Live Camera")

if start_camera:
    cap = cv2.VideoCapture(0)  # Try to open webcam (Works on PC)

    if not cap.isOpened():
        st.warning("Webcam not accessible! Using manual capture instead.")
        cap = None  # Disable OpenCV camera
    else:
        st.write("Camera is active!")

    stframe = st.empty()  # Placeholder for video stream

    while start_camera:
        if cap:  # If webcam is available, use OpenCV
            success, frame = cap.read()
            if not success:
                st.warning("Could not access the camera.")
                break
        else:
            camera_input = st.camera_input("Take a picture")  # Fallback for mobile
            if camera_input is not None:
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                continue  # Wait for image

        # Face Detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue
            
            processed_face = preprocess_image(face)
            prediction = model.predict(processed_face)[0][0]
            gender = "Male" if prediction < 0.5 else "Female"

            # Draw box & label
            color = (255, 0, 0) if gender == "Male" else (255, 20, 147)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display video frame
        stframe.image(frame, channels="BGR")

    if cap:
        cap.release()  # Release camera when done

# **2. Image Upload (Alternative for Mobile)**
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]
        gender = "Male" if prediction < 0.5 else "Female"

        st.write(f"Predicted Gender: **{gender}**")
