import streamlit as st

def show():
    st.title("🧪 Méthode")
    st.subheader("LoRa fine-tuning")

    st.write("Le fine-tuning LoRa a été utilisé pour adapter le modèle à la tâche spécifique de classification d'articles liés au climat. \n"
             "Cette approche permet d'ajuster les poids du modèle pré-entraîné tout en conservant la plupart de ses paramètres d'origine, ce qui le rend plus efficace et moins coûteux en ressources. \n"
             "Nous avons décidé d'utiliser des modèles entre 1.5 et 8 milliards de paramètres, car ils sont plus légers et peuvent être exécutés sur de petits GPUs et même sur google colab en chargeant les poids des modèles sur 4-bit ou 8-bit. \n"
             "L'objectif est de montrer que de petits LLMs fine-tunés peuvent atteindre, voir dépasser les performances de modèles plus grands et plus coûteux en ressources. \n")

    st.subheader("Catégories sélectionnées")
    st.write(
        "Nous avons examiné l'évolution temporelle de la couverture médiatique du climat, les thèmes les plus discutés selon les catégories ci-dessous."
    )
    cols = st.columns(2)
    with cols[0]:
        st.image("streamlit_app/images/lora.png", caption="LoRa fine-tuning", use_container_width='auto')
   
    with cols[1]:
        st.image("streamlit_app/images/Categories.png", caption="Nomenclature des sous-thématiques climat", use_container_width='auto')

   
    st.subheader("Données d'entraînement et inférence")
    st.write(
        "Un dataset de 141 articles annotés sur lequel le modèle précédent avait été entraîné nous a permis de sélectionner nos modèles et de calcluer les métriques. \n"
        "Cependant, nous avons décidé de ré-annoter un dataset de 196 données en raison de la qualité de l'annotation et de la diversité des articles. \n"
        "Nous avons donc utilisé ce dataset pour le fine-tuning du modèle finale (basé sur Distill-Llama-8B) et qui nous a permis de faire l'inférence sur les transcripts de TF1 et France 2 de l'année 2024 (16068 extraits)."
    )
