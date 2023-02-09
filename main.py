import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

model = load_model("data\ER_model_training12ep.h5")

face_cascade = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

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

    cv2.imshow('Emotion Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows