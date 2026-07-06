import streamlit as st
import cv2
import numpy as np

st.title("🔢 Contador de Objetos")
st.markdown("Sube una imagen (por ejemplo, monedas) y el sistema detectará los contornos y contará cuántos objetos hay.")

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Leer imagen subida
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_o = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Redimensionar para mostrar correctamente
    max_width = 800
    if image_o.shape[1] > max_width:
        ratio = max_width / image_o.shape[1]
        image_o = cv2.resize(image_o, (max_width, int(image_o.shape[0] * ratio)))
        
    image_gray = cv2.cvtColor(image_o, cv2.COLOR_BGR2GRAY)

    # Thresholding (Umbralización inversa)
    # Se expone el umbral en un slider para que el usuario pueda ajustarlo si la imagen es diferente
    umbral = st.slider("Ajuste de Umbral (Threshold)", 0, 255, 220)
    
    _, ther = cv2.threshold(image_gray, umbral, 255, cv2.THRESH_BINARY_INV)
    
    # Buscar contornos
    cntrs, _ = cv2.findContours(ther, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    moneda_num = 1
    objetos_validos = []

    area_minima = st.slider("Área mínima para ser considerado objeto (evitar ruido)", 0, 5000, 1000)

    for c in cntrs:
        if cv2.contourArea(c) < area_minima:
            continue
            
        objetos_validos.append(c)
            
        M = cv2.moments(c)
        if M["m00"] == 0: 
            M["m00"] = 1
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])

        mensaje = 'Num:' + str(moneda_num)
        cv2.putText(image_o, mensaje, (x-40, y), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawContours(image_o, [c], 0, (0, 255, 0), 3)
        
        moneda_num += 1

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original Procesada")
        st.image(image_o, channels="BGR", use_container_width=True)
        
    with col2:
        st.subheader("Máscara (Threshold)")
        st.image(ther, use_container_width=True)
        
    st.success(f"🎉 **¡Se encontraron {len(objetos_validos)} objetos!**")
