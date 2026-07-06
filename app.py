import streamlit as st

st.set_page_config(
    page_title="Proyectos AI Camara",
    page_icon="📷",
    layout="wide"
)

st.title(" Proyectos AI Camara - Menú Principal")

st.markdown("""
Bienvenido a la colección de proyectos de Inteligencia Artificial y Visión por Computadora.
Usa el menú lateral para navegar entre los diferentes mini proyectos y pruebas.

### Proyectos Principales
* **Generador de Poemas:** Analiza una imagen usando YOLO para detectar objetos, y genera un poema personalizado sobre ellos usando IA generativa.

### Mini Proyectos
* **Contador de Objetos:** Sube una imagen (ej. monedas) y usa OpenCV para contar cuántos objetos hay mediante la detección de contornos.
* **Detector de Caras y Ojos:** Toma una foto con tu cámara y usamos clasificadores Haar (Cascade) para detectar rostros y ojos.
* **Filtros de Color:** Selecciona un rango de color (Verde, Rojo, Amarillo, Azul) y procesamos la imagen para dejar solo ese color usando el espacio HSV.

---
*Selecciona un proyecto en la barra lateral para empezar.*
""")
