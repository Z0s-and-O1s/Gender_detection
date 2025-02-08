import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gendermodel.keras")

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

st.title("Live Gender Detection (WebRTC)")

# Video processing function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        if face.size == 0:
            continue
        
        processed_face = preprocess_image(face)
        prediction = model.predict(processed_face)[0][0]
        gender = "Male" if prediction < 0.5 else "Female"

        color = (255, 0, 0) if gender == "Male" else (255, 20, 147)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Stream
webrtc_streamer(
    key="gender-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
