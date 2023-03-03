import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import threading

# Model and cascade files
model_file = "data\ER_model_training12ep.h5"
cascade_file = "data\haarcascade_frontalface_default.xml"

# Load model and cascade classifier
model = load_model(model_file)
face_cascade = cv2.CascadeClassifier(cascade_file)

# Emotion labels and colors
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]

# Camera parameters
camera_index = 0
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
    cap.release()

if not available_cameras:
    print("No cameras available. Exiting.")
    exit()

print(f"Available cameras: {available_cameras}")
camera_input = int(input("Camera To Select: "))

if camera_index not in available_cameras:
    camera_index = int(input(f"Select a camera from {available_cameras}: "))

cap = cv2.VideoCapture(camera_input)

# Thread function to read frames from the camera
def read_camera():
    global cap, frame, ret
    while True:
        ret, frame = cap.read()
        if not ret:
            break

# Thread function to process frames for emotion detection
def detect_emotions():
    global frame
    while True:
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = roi_gray / 255.0
            emotion_probabilities = model.predict(roi_gray)
            emotion_index = np.argmax(emotion_probabilities[0])
            emotion = emotion_labels[emotion_index]
            cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_colors[emotion_index], 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_colors[emotion_index], 2)
        cv2.imshow('Emotion Detection ZSCL By KPZ', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads
frame = None
ret = False
camera_thread = threading.Thread(target=read_camera)
emotion_thread = threading.Thread(target=detect_emotions)
camera_thread.start()
emotion_thread.start()

# Wait for threads to finish
camera_thread.join()
emotion_thread.join()

cap.release()
cv2.destroyAllWindows()
