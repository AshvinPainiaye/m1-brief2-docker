import streamlit as st
import requests
import os

st.title("Prédiction")

age = st.number_input("Age")
revenu_estime_mois = st.number_input("Salaire")

API_URL = os.getenv("API_URL", "http://api:8000")

if st.button("Prediction"):
    response = requests.post(API_URL + "/predict", json={
        "age": age,
        "revenu_estime_mois": revenu_estime_mois
    })
    st.write(response.json())