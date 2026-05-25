from polars.datatypes import extension
import webbrowser
from typing import Counter
import imutils
import cv2
import numpy as np 
import streamlit as st
from ultralytics import YOLO
import webbrowser
from google import genai
from google.genai import types
import os

st.set_page_config(page_title="Poem App Generator", layout="centered")

modelo = YOLO("best_3.pt")
_, col_right = st.columns([5, 1])
my_slack = "https://hackclub.enterprise.slack.com/archives/D0ATFQU6B7C"
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
folder = "My poems"

if "poem" not in st.session_state:
        st.session_state.poem = ""

@st.dialog("Save Poem", width="large")
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

st.title("Welcome to my poem app generator")
st.markdown("*Pls upload ur image w the objects u want ur poem be about*")
st.divider()

image_uploaded=st.file_uploader("import poem image here :D", type=["jpg", "png", "jpeg"])

if image_uploaded is not None:

        objects =[]

        file_bytes = np.asarray(bytearray(image_uploaded.read()), dtype=np.uint8) #convertimos la imagen bytes
        proccesed_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)#lo transformamos en algo que yolo pueda leer
        resultados = modelo(proccesed_image)#aplicamos el modelo
        res_plotted = resultados[0].plot() #dibujamos las cajas
        
        st.subheader("Image Analysis")
        st.image(res_plotted, channels="BGR", width=640) #mostramos la imagenes

        deteccion=resultados[0]

        for d in deteccion.boxes:
                id_num=d.cls.item()
                id_num=int(id_num)
                print(id_num)
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

        if st.button("Generate Poem"):

                if style =="67, no cap, skibidi, sigma, pookie":
                    style = "Use a super gen z terminology, based on tiktok/instagrams/reddit slag, ignore a lil bit the style of the author and focus on be hilarious, integrate the slang without sound forced"

                if length == "Small":
                        style = "No more than 100 words"
                        config = types.GenerateContentConfig(max_output_tokens=500)
                elif length == "Medium":
                        style = "No more than 200 words"
                        config = types.GenerateContentConfig(max_output_tokens=1000)
                elif length == "Large (pay for the tokens >:c)":
                        style= "No more than 300 words"
                        config = types.GenerateContentConfig(max_output_tokens=1500)

                if objects == {}:
                        objects == "Talk about the empty and lonely feeling of nothing"

                poem_generation=client.models.generate_content(
                config=config, 
                model="gemini-2.5-flash-lite", 
                contents=f"""
                [ROLE AND STYLE]
                Actúa como un escritor experto en literatura. Escribe un poema con un estilo "{style}" del autor "{autor}". 
                El poema debe capturar la atmósfera, el tono y la esencia característica de este autor.

                [INPUT DATA]
                Debes incluir e integrar de forma fluida los siguientes elementos detectados en la escena:
                {objects}

                [CONSTRAINTS]
                1. OUTPUT LANGUAGE: Escribe el poema única y exclusivamente en IDIOMA INGLÉS.
                2. LENGTH: El largo del poema debe ser "{length}" (Máximo 500 palabras).
                3. FORMAT: Entrega el poema directamente. Está ESTRICTAMENTE PROHIBIDO incluir introducciones, saludos, comentarios aclaratorios o conclusiones (ej. No escribas "Here is your poem:"). Inicia inmediatamente con el primer verso.
                """)
                
                st.session_state.poem = poem_generation

        # Display the poem if it exists in session state
        if st.session_state.poem != "":
                with st.container(border=True):
                        st.markdown(st.session_state.poem)

        st.divider()
        if st.session_state.poem != "":
                if st.button("Save as txt"):
                        save_poem_dialog(st.session_state.poem)


with col_right:
        st.button("DM on slack", on_click=webbrowser.open, args=(my_slack,), width=100)
