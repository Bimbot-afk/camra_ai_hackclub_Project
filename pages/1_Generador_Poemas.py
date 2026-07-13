import streamlit as st
import webbrowser
from collections import Counter
import cv2
import numpy as np 
from ultralytics import YOLO
import requests
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

st.markdown("Consigue tu API Key en: [https://ai.hackclub.com/dashboard](https://ai.hackclub.com/dashboard)")
user_api_key = st.text_input("Ingresa tu Hack Club AI API Key:", type="password")

modelo_ai = st.selectbox("Selecciona el modelo AI:", [
    "tencent/hy3:free",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-luna-pro",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-terra-pro",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-sol-pro",
    "x-ai/grok-4.5",
    "~x-ai/grok-latest",
    "aion-labs/aion-3.0-mini",
    "aion-labs/aion-3.0",
    "tencent/hy3"
])

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

with st.expander("🔍 Ver objetos que el modelo puede detectar"):
    st.markdown("""
    El modelo está entrenado para reconocer los siguientes objetos:
    * Gorra (`cap`)
    * Carro Colombiano (`colombian_car`)
    * Mono Grande (`monkey_big`)
    * Mono Pequeño (`monkey_small`)
    * Lata de Monster (`monster_can`)
    * Otra Lata (`not_monster_can`)
    * Lápiz (`pencil`)
    * Control de PS5 (`ps5_contorller`)
    * Oso de Peluche (`teddy_bear`)
    """)

st.divider()

# Usage limit warning
uses_left = MAX_USES - st.session_state.generations_count
st.info(f"Tienes {uses_left} generaciones de poemas disponibles en esta sesión.")

if not user_api_key:
    st.warning("⚠️ Ingresa tu API Key arriba para poder generar poemas.")

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

    # Disable button if limit reached or API key missing
    button_disabled = st.session_state.generations_count >= MAX_USES or not user_api_key

    if st.button("Generate Poem", disabled=button_disabled):
        
        st.session_state.generations_count += 1

        if style =="67, no cap, skibidi, sigma, pookie (brainrot)":
            style = "Use a super gen z terminology, based on tiktok/instagrams/reddit slag, ignore a lil bit the style of the author and focus on be hilarious, integrate the slang without sound forced"

        if length == "Small":
            style_length = "No more than 100 words"
            max_tokens = 500
        elif length == "Medium":
            style_length = "No more than 200 words"
            max_tokens = 1000
        elif length == "Large (pay for the tokens >:c)":
            style_length = "No more than 300 words"
            max_tokens = 1500

        # Fix empty objects bug from original code
        if not objects:
            objects_str = "Talk about the empty and lonely feeling of nothing"
        else:
            objects_str = str(objects)

        prompt_text = f"""
[ROLE]

You are an award-winning poet.

Write a poem in the style of "{style}", strongly inspired by the literary voice of "{autor}".
Capture the author's rhythm, imagery, emotional depth and worldview without quoting or copying their original works.

[SCENE]

The following objects are ALL present in front of the observer.

{objects_str}

These objects are NOT decorations.
They ARE the source of the poem.

Imagine they exist together in the same physical place.
Discover relationships between them.
Give symbolic meaning to them.
Allow them to shape the emotion, narrative and imagery.

If the detected objects are playful, the poem should naturally become playful.
If they are lonely, the poem should become melancholic.
If they create contrast, build the poem around that contrast.

The poem should feel impossible to write if these specific objects were replaced with different ones.

Do not simply list the objects.
Instead, transform them into metaphors, symbols, memories or characters.

Every detected object must influence at least one image or idea.

[OUTPUT]

Language: English only.

Length: {style_length}

Maximum 500 words.

Output ONLY the poem.

Begin immediately with the first line.
"""

        with st.spinner("Pensando el poema..."):
            try:
                from openrouter import OpenRouter
                client = OpenRouter(
                    api_key=user_api_key,
                    server_url="https://ai.hackclub.com/proxy/v1"
                )
                
                response = client.chat.send(
                    model=modelo_ai,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=max_tokens,
                    timeout_ms=25000  # 25 seconds timeout to prevent freezing
                )
                
                # Check response format
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    st.session_state.poem = response.choices[0].message.content
                else:
                    st.error("La API devolvió una respuesta vacía o con error.")
                    st.session_state.generations_count -= 1
            except Exception as e:
                st.error(f"Error conectando con la IA (¿Se agotó el tiempo de espera?): {e}")
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
