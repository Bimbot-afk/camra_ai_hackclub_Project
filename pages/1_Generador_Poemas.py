import streamlit as st
import webbrowser
from collections import Counter
import cv2
import numpy as np 
from ultralytics import YOLO
from google import genai
from google.genai import types
import os

st.title("Generador de Poemas AI ✍️")

# Add usage limit logic
MAX_USES = 3
if "generations_count" not in st.session_state:
    st.session_state.generations_count = 0

if "poem" not in st.session_state:
    st.session_state.poem = ""

# Load model, updated path since we are in root directory now
@st.cache_resource
def load_yolo():
    # Attempt to load the model from the known path
    model_path = os.path.join("models", "best_3.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

modelo = load_yolo()

if not modelo:
    st.error("Error: No se encontró el modelo YOLO en models/best_3.pt")
    st.stop()

my_slack = "https://hackclub.enterprise.slack.com/archives/D0ATFQU6B7C"

# Safe access to secrets
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("No se encontró la GEMINI_API_KEY en los secretos. Configúrala en .streamlit/secrets.toml")
    st.stop()

folder = "My poems"

@st.dialog("Guardar Poema", width="large")
def save_poem_dialog(poem_text):
    st.write("Make sure to put a cool title >:)")
    title = st.text_input("Title:")
    if st.button("Confirm and Save"):
        if not title.strip():
            st.error("Title cannot be empty!")
            return
        
        filename = title.strip()
        if not filename.endswith(".txt"):
            filename += ".txt"
        
        route = os.path.join(folder, filename)
        if os.path.exists(route):
            st.error("Looks like this name already exists")
        else:
            try:
                with open(route, "w", encoding="utf-8") as archivo:
                    archivo.write(poem_text)
                st.success(f"Poem saved as {filename}!")
            except Exception as e:
                st.error(f"Error saving poem: {e}")

if not os.path.exists(folder):
    os.mkdir(folder)

st.markdown("*Por favor sube tu imagen con los objetos sobre los que quieres que trate el poema*")
st.divider()

# Usage limit warning
uses_left = MAX_USES - st.session_state.generations_count
st.info(f"Tienes {uses_left} generaciones de poemas disponibles en esta sesión.")

image_uploaded=st.file_uploader("import poem image here :D", type=["jpg", "png", "jpeg"])

if image_uploaded is not None:
    objects =[]

    file_bytes = np.asarray(bytearray(image_uploaded.read()), dtype=np.uint8) 
    proccesed_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    resultados = modelo(proccesed_image)
    res_plotted = resultados[0].plot() 
    
    st.subheader("Image Analysis")
    st.image(res_plotted, channels="BGR", width=640)

    deteccion=resultados[0]

    for d in deteccion.boxes:
        id_num=d.cls.item()
        id_num=int(id_num)
        item_name= deteccion.names[id_num]
        objects.append(item_name)

    c_objects=Counter(objects)
    
    st.divider()
    st.subheader("Detected Objects")
    st.write("below the object my model detected")
    st.write(c_objects)

    st.divider()
    st.write("Personalization")

    autor= st.radio("Choose ur favorite colombian author:", ("Gabriel Garcia Marques", "Mario Mendoza", "Rafael Pombo"))
    style=st.radio("Choose ur favorite style:", ("Magic realism", "Melancholy", "Kids", "Happy", "67, no cap, skibidi, sigma, pookie (brainrot)"))
    length=st.radio("Choose ur length:", ("Small", "Medium", "Large (pay for the tokens >:c)"))
    
    st.divider()
    st.subheader("Your Poem")

    # Disable button if limit reached
    button_disabled = st.session_state.generations_count >= MAX_USES

    if st.button("Generate Poem", disabled=button_disabled):
        
        st.session_state.generations_count += 1

        if style =="67, no cap, skibidi, sigma, pookie":
            style = "Use a super gen z terminology, based on tiktok/instagrams/reddit slag, ignore a lil bit the style of the author and focus on be hilarious, integrate the slang without sound forced"

        if length == "Small":
            style_length = "No more than 100 words"
            config = types.GenerateContentConfig(max_output_tokens=500)
        elif length == "Medium":
            style_length = "No more than 200 words"
            config = types.GenerateContentConfig(max_output_tokens=1000)
        elif length == "Large (pay for the tokens >:c)":
            style_length = "No more than 300 words"
            config = types.GenerateContentConfig(max_output_tokens=1500)

        # Fix empty objects bug from original code
        if not objects:
            objects_str = "Talk about the empty and lonely feeling of nothing"
        else:
            objects_str = str(objects)

        with st.spinner("Pensando el poema..."):
            try:
                response = client.models.generate_content(
                config=config, 
                model="gemini-2.5-flash-lite", 
                contents=f"""
                [ROLE AND STYLE]
                Actúa como un escritor experto en literatura. Escribe un poema con un estilo "{style}" del autor "{autor}". 
                El poema debe capturar la atmósfera, el tono y la esencia característica de este autor.

                [INPUT DATA]
                Debes incluir e integrar de forma fluida los siguientes elementos detectados en la escena:
                {objects_str}

                [CONSTRAINTS]
                1. OUTPUT LANGUAGE: Escribe el poema única y exclusivamente en IDIOMA INGLÉS.
                2. LENGTH: El largo del poema debe ser "{style_length}" (Máximo 500 palabras).
                3. FORMAT: Entrega el poema directamente. Está ESTRICTAMENTE PROHIBIDO incluir introducciones, saludos, comentarios aclaratorios o conclusiones (ej. No escribas "Here is your poem:"). Inicia inmediatamente con el primer verso.
                """)
                st.session_state.poem = response.text
            except Exception as e:
                st.error(f"Error generando poema: {e}")
                st.session_state.generations_count -= 1 # Revert count on error

    # Display the poem if it exists in session state
    if st.session_state.poem != "":
        with st.container(border=True):
            st.markdown(st.session_state.poem)

    st.divider()
    if st.session_state.poem != "":
        if st.button("Save as txt"):
            save_poem_dialog(st.session_state.poem)

_, col_right = st.columns([5, 1])
with col_right:
    st.link_button("DM on slack", my_slack)
