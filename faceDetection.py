import numpy as np
import cv2
import os

# 1. Creating haar cascades
face_cascade = cv2.CascadeClassifier('haarCascade/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('haarCascade/haarcascade_profileface.xml')
eyes_cascade = cv2.CascadeClassifier('haarCascade/haarcascade_eye.xml')

# 2. Upload external video stream - in the end it will be video from security cameras.
cap = cv2.VideoCapture('facesVideo.avi')

# 3. For each frame in the video, try to detect faces using the detectMultiScale() function.
# if a face/eyes detected - save the crop image of the face in 'testImages' directory.
currentFrame = 0
while cap.isOpened():
    ret, frame = cap.read()  # capture frame-by-frame

    if not ret:
        break
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        img = frame[y: end_cord_y, x:end_cord_x]
        name = './testImages/face' + str(currentFrame) + '.jpg'
        cv2.imwrite(name, img)
        currentFrame += 1

    profileFaces = profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in profileFaces:
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        img = frame[y: end_cord_y, x:end_cord_x]
        name = './testImages/profileFace' + str(currentFrame) + '.jpg'
        cv2.imwrite(name, img)
        currentFrame += 1

    cv2.imshow('frame', frame)

