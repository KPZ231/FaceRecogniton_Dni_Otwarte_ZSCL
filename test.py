import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import threading as tr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

width = int(input("Select Width Of Screen: "))
height = int(input("Select Height Of Screen: "))

# ░█░█░█▀█░▀▀█
# ░█▀▄░█▀▀░▄▀░
# ░▀░▀░▀░░░▀▀▀`
# region Modele Danych

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Konfiguracja dla dostępnej karty graficznej
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=1024 * 4)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Błąd konfiguracji
        print(e)


model = tf.keras.models.load_model("data\ER_model_training12ep.h5")

face_cascade = cv2.CascadeClassifier(
    "data\haarcascade_frontalface_default.xml")


# endregion
# region Wczytywanie Danych
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
                  (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255)]
# endregion

# paramety


# region camera index
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
camera_input = int(input(" Camera To Select: "))

if camera_index not in available_cameras:
    camera_index = int(input(f"Select a camera from {available_cameras}: "))


# endregion

cap = cv2.VideoCapture(camera_input)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def main():
    frame_count = 0
    start_time = cv2.getTickCount()
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (10, 30)
    font_scale = 0.5
    font_color = (0, 255, 0)
    line_type = 1

    while True:
        ret, frame = cap.read()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cv2.namedWindow("Emotion Detection ZSCL By KPZ")
        cv2.resizeWindow("Emotion Detection ZSCL By KPZ", width, height)

        if ret:
            frame_count += 1

            elapsed_time = (cv2.getTickCount() - start_time) / \
                cv2.getTickFrequency()
            current_fps = int(frame_count / elapsed_time)

            fps_text = f"FPS: {current_fps}"
            cv2.putText(frame, fps_text, pos, font,
                        font_scale, font_color, line_type)
        else:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            roi_gray = roi_gray / 255.0

            emotion_probabilities = model.predict(roi_gray)
            emotion_index = np.argmax(emotion_probabilities[0])
            emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          emotion_colors[emotion_index], 1)

            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, emotion_colors[emotion_index], 1)

        cv2.imshow('Emotion Detection ZSCL By KPZ', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


thread = tr.Thread(target=main)
thread.start()
thread.join()

cap.release()

cv2.destroyAllWindows
