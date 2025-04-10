import streamlit as st
import my_pages.dashboard_2024 as dashboard
import my_pages.methode as methode
import my_pages.comparaison_modeles as comparaison
import my_pages.Sous_categorie as sous_categorie


# Fonction pour afficher le logo
def add_logo():
    st.markdown("""
        <style>
        .logo {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            border-radius: 0 !important;  /* Enlever les bords arrondis */
            border: none !important;      /* Enlever toute bordure */
            padding: 0 !important;        /* Supprimer le padding si appliqué */
        }
        img {
            border-radius: 0 !important;  /* Réinitialiser tous les bords arrondis de l'image */
            border: none !important;      /* Réinitialiser toute bordure */
        }
        </style>
    """, unsafe_allow_html=True)
    st.image("streamlit_app/images/logo.png", width=100, use_container_width=False)

# Configurer la page
st.set_page_config(page_title="Baroclima", layout="wide", initial_sidebar_state="collapsed")

# Ajouter le logo sur toutes les pages
add_logo()


# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisissez une page :",
    ["Observatoire JT 2024", "Sous-classification JT 2024", "Méthode", "Comparaison des modèles"]
)

# Chargement dynamique des pages
if page == "Observatoire JT 2024":
    dashboard.show()

elif page == "Méthode":
    methode.show()

elif page == "Comparaison des modèles":
    comparaison.show()

elif page == "Sous-classification JT 2024":
    sous_categorie.show()