from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load trained model
model = tf.keras.models.load_model("gendermodel.h5")

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Image Preprocessing
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image

# **1. Webcam Stream**
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect Face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue  # Skip if face detection fails

            processed_face = preprocess_image(face)

            # Predict Gender
            prediction = model.predict(processed_face)[0][0]
            gender = "Male" if prediction < 0.5 else "Female"

            # Draw Box & Label
            color = (255, 0, 0) if gender == "Male" else (255, 20, 147)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Send frame to webpage
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# **2. Video Stream Route**
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# **3. Image Upload Route**
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process Image Properly
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Invalid image file"})

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    processed_image = preprocess_image(image)

    # Predict Gender
    prediction = model.predict(processed_image)[0][0]
    gender = "Male" if prediction < 0.5 else "Female"

    return jsonify({"gender": gender, "filepath": filepath})

# **4. Home Route**
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
