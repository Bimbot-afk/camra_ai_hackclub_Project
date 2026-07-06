import streamlit as st
import cv2
import numpy as np

st.title("🧑 Detector de Caras y Ojos")
st.markdown("¡Toma una foto con tu cámara web y el sistema detectará tu rostro y ojos!")

# Cargar los clasificadores (se cargan una sola vez con caché)
@st.cache_resource
def load_classifiers():
    face_clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_clf, eye_clf

faceClasification, eyeClasification = load_classifiers()

picture = st.camera_input("Toma una foto")

if picture:
    # Convertir la foto tomada a formato OpenCV
    file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # El st.camera_input ya suele mostrarse como espejo, pero si necesitamos voltearlo internamente:
    # frame = cv2.flip(frame, 1) # Dependiendo del navegador puede no ser necesario
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detección de caras
    faces = faceClasification.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 10))
    
    for (x, y, w, h) in faces:
        # Cuadrado verde para la cara
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Los ojos se buscan SOLO dentro de la cara
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eyeClasification.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes:
            # Cuadrado azul para los ojos
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    st.subheader("Resultado")
    if len(faces) == 0:
        st.warning("No se detectó ningún rostro.")
    else:
        st.success(f"Se detectaron {len(faces)} rostro(s).")
        
    st.image(frame, channels="BGR", use_column_width=True)
