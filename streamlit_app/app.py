import streamlit as st
import my_pages.dashboard_2024 as dashboard
import my_pages.methode as methode
import my_pages.comparaison_modeles as comparaison
import my_pages.Sous_categorie as sous_categorie

st.set_page_config(page_title="Baroclima", layout="wide", initial_sidebar_state="collapsed")


# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une page :",
    ["Dashboard 2024", "Méthode", "Comparaison des modèles", "Sous-catégorie"]
)

# Chargement dynamique des pages
if page == "Dashboard 2024":
    dashboard.show()

elif page == "Méthode":
    methode.show()

elif page == "Comparaison des modèles":
    comparaison.show()

elif page == "Sous-catégorie":
    sous_categorie.show()