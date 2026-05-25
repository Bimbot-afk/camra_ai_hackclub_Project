import webbrowser
from typing import Counter
import imutils
import cv2
import numpy as np 
import streamlit as st
from ultralytics import YOLO
import requests
import json

st.set_page_config(page_title="Poem App Generator", layout="centered")

modelo = YOLO("best_3.pt")
_, col_right = st.columns([5, 1])
my_slack = "https://hackclub.enterprise.slack.com/archives/D0ATFQU6B7C"
api_key = st.secrets["LLAMA_API_KEY"]

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
        style=st.radio("Choose ur favorite style:", ("Magic realism", "Melancholy", "Kids", "Happy", "67, no cap, skibidi, sigma, pookie"))
        length=st.radio("Choose ur length:", ("Small", "Medium", "Large (pay for the tokens >:c)"))
        
        st.divider()
        st.subheader("Your Poem")

        if st.button("Generate Poem"):
                
                system_instruction = """
                You are an expert literary writer. Write a poem that strictly imitates the style of the specified author and style.
                Integrate the provided objects seamlessly into the narrative or atmosphere of the poem.

                CONSTRAINTS:
                1. OUTPUT LANGUAGE: Write the poem ONLY and exclusively in ENGLISH.
                2. LENGTH: Adhere to the requested length. Maximum 500 words.
                3. FORMAT: Output the poem directly. Do NOT include any introductions, greetings, or meta-commentary (e.g., Do NOT write "Here is your poem:"). Start immediately with the first verse.
                """

                user_data = f"""
                Author: {autor}
                Style: {style}
                Length: {length}
                Objects detected by YOLO: {objects}
                """
                if style =="67, no cap, skibidi, sigma, pookie":
                    style = "Use a super gen z terminology, based on tiktok/instagrams/reddit slag, ignore a lil bit the style of the author and focus on be hilarious"


                api_request_json = {
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_data}
                    ]
                }
                
                # 2 llamada optimizada al modelo usando OpenRouter
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=api_request_json
                )

                with st.container(border=True):
                    if response.status_code == 200:
                        poem_text = response.json()["choices"][0]["message"]["content"]
                        st.markdown(poem_text)
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

with col_right:
        st.button("DM on slack", on_click=webbrowser.open, args=(my_slack,), width=100)