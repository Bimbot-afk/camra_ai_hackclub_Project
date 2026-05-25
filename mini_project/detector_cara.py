import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

faceClasification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeClasification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClasification.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 10, 0))

    for (x, y, w, h) in faces:
        # Cuadrado verde para la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Los ojos se buscan SOLO dentro de la cara para mejorar rendimiento y evitar falsos positivos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyeClasification.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()