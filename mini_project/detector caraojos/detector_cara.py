import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Se agrega cv2.data.haarcascades para asegurar que encuentre los archivos del modelo
faceClasification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeClasification = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mejoras: scaleFactor=1.2 y minNeighbors=7 (más estricto con falsos positivos)
    # minSize=(70,70) ignora recuadros pequeños para que no detecte caras falsas en el fondo.
    faces = faceClasification.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Cuadrado verde para la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # --- PREPARACIÓN PARA OJOS ---
        # Los ojos se buscan SOLO dentro de la cara para mejorar rendimiento y evitar falsos positivos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Descomenta las siguientes 3 líneas para activar la detección de ojos (cuadrado azul BGR: 255, 0, 0)
        eyes = eyeClasification.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()