import streamlit as st
import cv2
import numpy as np

st.title("🎨 Detector de Colores")
st.markdown("¡Sube una foto o tómate una captura y filtra los colores (Verde, Rojo, Amarillo, Azul)!")

def dibujar_contorno(frame, mask, color):
    # Mejoras para reducir ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0: 
                M["m00"] = 1
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
            
            cv2.circle(frame, (x,y), 7, color, -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, '{},{}'.format(x,y), (x+10,y), font, 0.75, color, 1, cv2.LINE_AA)
            
            nuevo_contorno = cv2.convexHull(c)
            cv2.drawContours(frame, [nuevo_contorno], 0, color, 3)
            
    return frame

# Definir rangos de color
rangos = {
    "Verde": {
        "bajo": np.array([45, 50, 20], np.uint8),
        "alto": np.array([75, 255, 255], np.uint8),
        "color_bgr": (0, 255, 0)
    },
    "Rojo 1": { # El rojo tiene dos rangos en HSV
        "bajo": np.array([0, 100, 20], np.uint8),
        "alto": np.array([10, 255, 255], np.uint8),
        "color_bgr": (0, 0, 255)
    },
    "Rojo 2": {
        "bajo": np.array([165, 100, 20], np.uint8),
        "alto": np.array([179, 255, 255], np.uint8),
        "color_bgr": (0, 0, 255)
    },
    "Amarillo": {
        "bajo": np.array([25, 100, 20], np.uint8),
        "alto": np.array([35, 255, 255], np.uint8),
        "color_bgr": (0, 255, 255)
    },
    "Azul": {
        "bajo": np.array([90, 50, 20], np.uint8),
        "alto": np.array([125, 230, 255], np.uint8),
        "color_bgr": (255, 0, 0)
    }
}

opcion = st.radio("¿Qué deseas usar?", ("Subir Imagen", "Cámara"))

picture = None
if opcion == "Subir Imagen":
    picture = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])
else:
    picture = st.camera_input("Toma una foto")

color_elegido = st.selectbox("Selecciona el color a detectar", ["Verde", "Rojo", "Amarillo", "Azul", "Todos"])

if picture:
    file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Suavizamos la imagen (Blur)
    frame_suavizado = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
    
    # Crear un diccionario para las máscaras
    mascaras = {}
    
    for nombre, rango in rangos.items():
        mascaras[nombre] = cv2.inRange(frame_hsv, rango["bajo"], rango["alto"])
        
    if color_elegido == "Rojo":
        # Combinar las dos máscaras rojas
        mascara_final = cv2.add(mascaras["Rojo 1"], mascaras["Rojo 2"])
        frame = dibujar_contorno(frame, mascara_final, rangos["Rojo 1"]["color_bgr"])
    elif color_elegido == "Todos":
        for nombre, rango in rangos.items():
            frame = dibujar_contorno(frame, mascaras[nombre], rango["color_bgr"])
    else:
        frame = dibujar_contorno(frame, mascaras[color_elegido], rangos[color_elegido]["color_bgr"])

    st.image(frame, channels="BGR", use_column_width=True)
