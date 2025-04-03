import streamlit as st

st.title("Ma Première Application Streamlit")
st.write("Bienvenue dans votre application interactive!")

# Recueillir une saisie utilisateur
nom = st.text_input("Entrez votre nom :")

if nom:
    st.write("Bonjour", nom, "!")
