import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
from threading import Thread
import time


#░█░█░█▀█░▀▀█
#░█▀▄░█▀▀░▄▀░
#░▀░▀░▀░░░▀▀▀

#region Modele Danych

model = load_model("data\ER_model_training12ep.h5")

face_cascade = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml")

#endregion 
#region Wczytywanie Danych
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]
#endregion

#paramety
frame_count = 0
start_time = cv2.getTickCount()
font = cv2.FONT_HERSHEY_SIMPLEX
pos = (10, 30)
font_scale = 0.5
font_color = (0, 255, 0)
line_type = 2


#region camera index
camera_index = 0
available_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
    cap.release()


if not available_cameras:
    print("No cameras available. Exiting.")
    time.sleep(5)
    exit()

print(f"Available cameras: {available_cameras}")
camera_input = int(input(" Camera To Select: "))

if camera_index not in available_cameras:
    camera_index = int(input(f"Select a camera from {available_cameras}: "))


#endregion

cap = cv2.VideoCapture(camera_input)

class _main(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.start()
    def run(self):
        while True:
            ret, frame = cap.read()

            #cv2.putText(frame, "Emotion Detection By KPZ", (00, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 0.4, cv2.LINE_AA, False)

            if ret:
                frame_count += 1

                elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                current_fps = int(frame_count / elapsed_time)

                fps_text = f"FPS: {current_fps}"
                cv2.putText(frame, fps_text, pos, font, font_scale, font_color, line_type)
            else:
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


main = _main()
main.run()
cap.release()

while True:
    pass

cv2.destroyAllWindows