import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns



def show():
    st.title("📊 Dashboard 2024")
    st.write("Bienvenue sur la page du Dashboard.")

    # Initialisation de la sélection dans session_state
    if "selection" not in st.session_state:
        st.session_state.selection = None
    # Disposition en colonnes pour avoir deux boutons côte à côte
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Micro"):
            st.session_state.selection = "Micro"
    with col2:
        if st.button("Macro"):
            st.session_state.selection = "Macro"

    # Initialisation
    csv_path = None
    categories = None

    if st.session_state.selection == "Micro":

        csv_path = "Micro_category/Inference/predictions_2024_Micro-lora-8B.csv"
        categories = [
            "gaz_effet_de_serre",
            "agriculture_et_utilisation_du_sol",
            "peche_et_chasse",
            "pollution",
            "deforestation",
            "surconsommation",
            "catastrophes_naturelles",
            "rechauffement_climatique_canicule",
            "secheresse",
            "couche_ozone",
            "feu_foret",
            "tension_alim_famines",
            "eau_potable",
            "hausse_niveau_mer_fonte_glace",
            "consequence_sociale",
            "acidification_ocean",
            "perte_biodiversite",
            "pollution",
            "energies_renouvelables_et_nucleaires",
            "transport_decarbone",
            "engagement_politique_et_entreprises",
            "activisme_eco",
            "solution_innovante",
            "comportement_consommateur",
            "reforestation"
        ]

    elif st.session_state.selection == "Macro":
        csv_path = "Macro_category/Inference/predictions_Macro-lora-8B-1-cat.csv"
        categories = [
            "pollution",
            "ecosystemes",
            "dereglement_climatique",
            "energie",
            "politique"
        ]

    # Charger le CSV si un bouton a été cliqué
    if csv_path and categories:

        # Chemin vers le fichier CSV
        df = pd.read_csv(csv_path)

        # Conversion de la colonne 'date' en datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Dictionnaire pour mapper les numéros de mois aux mois en français
        month_dict = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }

        # Création de la colonne 'mois' à partir de la date
        df["mois"] = df["date"].dt.month.map(month_dict)

        # ======================================================
        # 1. Affichage des filtres en haut de la page
        # ======================================================
        st.markdown("## Filtres")

        # Filtrer par mois
        mois_disponibles = df["mois"].dropna().unique().tolist()
        if mois_disponibles:
            mois_disponibles.sort(key=lambda x: list(month_dict.values()).index(x))
        selected_month = st.selectbox("Sélectionnez le mois", options=["Tous"] + mois_disponibles)

        # Filtrer par média
        medias_disponibles = df["media"].dropna().unique().tolist()
        selected_media = st.selectbox("Sélectionnez un média", options=["Tous"] + medias_disponibles)

        # ======================================================
        # 2. Application des filtres sur le DataFrame
        # ======================================================
        df_filtre = df.copy()
        # Application du filtre média
        if selected_media != "Tous":
            df_filtre = df_filtre[df_filtre["media"] == selected_media]
        # Application du filtre mois
        if selected_month != "Tous":
            df_filtre = df_filtre[df_filtre["mois"] == selected_month]

        # Suppression des colonnes non désirées
        colonnes_a_supprimer = [
            "order", "presenter", "editorDeputy", "url",
            "urlTvNews", "containsWordGlobalWarming", "top_words", 
            "confidence", "prediction_label"
        ]
        df_filtre.drop(columns=[col for col in colonnes_a_supprimer if col in df_filtre.columns], inplace=True)

        # ======================================================
        # 3. Filtre par catégorie (présente dans l'une des 3 colonnes)
        # ======================================================
        cat_cols = ["prediction_label_1", "prediction_label_2", "prediction_label_3"]
        all_cats = categories
        all_cats.sort()
        selected_cats = st.multiselect("Sélectionnez la ou les catégories", options=all_cats)

        # Appliquer le filtre catégorie si une sélection a été faite
        if selected_cats:
            df_filtre = df_filtre[
                df_filtre[cat_cols].apply(lambda row: any(cat in selected_cats for cat in row), axis=1)
            ]

        # Filtrer pour ne garder que les lignes dont les prédictions sont dans la liste "categories"
        df_filtre = df_filtre[
            df_filtre[cat_cols].apply(lambda row: all((cat in categories) if pd.notna(cat) else True for cat in row), axis=1)
        ]

        # ======================================================
        # 4. Pagination : Affichage de 100 lignes par page
        # ======================================================
        rows_per_page = 100
        total_rows = df_filtre.shape[0]
        total_pages = max(math.ceil(total_rows / rows_per_page), 1)

        st.markdown("## Tableau de données")
        st.write(f"Nombre total de lignes filtrées : {total_rows}")

        if "page" not in st.session_state:
            st.session_state.page = 1

        cols = st.columns([1, 2, 1])
        with cols[0]:
            if st.button("⬅️"):
                if st.session_state.page > 1:
                    st.session_state.page -= 1
        with cols[1]:
            st.markdown(f"<h5 style='text-align: center;'>Page {st.session_state.page} sur {total_pages}</h5>", unsafe_allow_html=True)
        with cols[2]:
            if st.button("➡️"):
                if st.session_state.page < total_pages:
                    st.session_state.page += 1

        start_idx = (st.session_state.page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        df_page = df_filtre.iloc[start_idx:end_idx]
        st.dataframe(df_page)

        # ======================================================
        # 5. Répartition des catégories prédites
        # ======================================================
        st.markdown("## Répartition des catégories prédites")
        all_preds = pd.concat([
            df_filtre["prediction_label_1"],
            df_filtre["prediction_label_2"],
            df_filtre["prediction_label_3"]
        ])
        all_preds = all_preds[all_preds.isin(categories)]
        cat_counts = all_preds.value_counts().sort_values(ascending=False)
        pastel_palette = sns.color_palette("pastel", len(cat_counts))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Histogramme vertical")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax1, palette=pastel_palette)
            ax1.set_xlabel("Catégorie")
            ax1.set_ylabel("Occurrences")
            ax1.set_title("Occurrences par catégorie")
            ax1.tick_params(axis='x', rotation=90)
            st.pyplot(fig1)
        with col2:
            st.markdown("### 🥧 Camembert")
            fig2, ax2 = plt.subplots(figsize=(7, 6))
            total = cat_counts.sum()
            percentages = (cat_counts / total * 100).round(1)
            labels_with_percent = [f"{cat} ({pct}%)" for cat, pct in zip(cat_counts.index, percentages)]
            wedges, _ = ax2.pie(
                cat_counts.values,
                startangle=90,
                colors=sns.color_palette("pastel", len(cat_counts)),
                wedgeprops=dict(width=0.6)
            )
            ax2.axis("equal")
            ax2.set_title("Répartition des catégories")
            ax2.legend(
                wedges,
                labels_with_percent,
                title="Catégories",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize="small"
            )
            st.pyplot(fig2)
        # ======================================================
        # 6. Histogramme proportionnel par mois avec ordre et palette fixes
        # ======================================================
        st.markdown("## Proportion des sous-catégories pour un mois sélectionné")

        # Slider pour sélectionner un mois (1 à 12)
        selected_month_single = st.slider("Sélectionnez un mois", min_value=1, max_value=12, value=1, step=1)
        selected_month_name = month_dict[selected_month_single]
    
        # Filtrer le DataFrame pour le mois sélectionné
        df_month = df_filtre[df_filtre["mois"] == selected_month_name]

        if df_month.empty:
            st.write("Aucune donnée pour ce mois.")
        else:
            # Combiner les prédictions des 3 colonnes pour le mois sélectionné
            all_preds_month = pd.concat([
                df_month["prediction_label_1"],
                df_month["prediction_label_2"],
                df_month["prediction_label_3"]
            ])
            # Ne conserver que les prédictions qui figurent dans la liste des catégories
            all_preds_month = all_preds_month[all_preds_month.isin(categories)]
            
            total_preds = all_preds_month.shape[0]
            if total_preds == 0:
                st.write("Aucune prédiction valide pour ce mois.")
            else:
                # Définir un ordre fixe pour les catégories (ordre alphabétique ici, à adapter si nécessaire)
                ordered_categories = sorted(categories)
                
                # Calculer le nombre d'occurrences par catégorie en réindexant pour avoir toutes les catégories
                cat_counts_month = all_preds_month.value_counts().reindex(ordered_categories, fill_value=0)
                proportions = cat_counts_month / total_preds * 100

                # Création d'une palette de couleurs fixe pour chaque catégorie
                palette = sns.color_palette("pastel", len(ordered_categories))
                color_mapping = dict(zip(ordered_categories, palette))
                colors = [color_mapping[cat] for cat in ordered_categories]
                
                # Création de l'histogramme
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x=ordered_categories, 
                    y=proportions.values, 
                    palette=colors, 
                    ax=ax
                )
                ax.set_xlabel("Catégorie")
                ax.set_ylabel("Pourcentage (%)")
                ax.set_title(f"Proportion des sous-catégories en {month_dict[selected_month_single]}")
                ax.tick_params(axis='x', rotation=90)
                st.pyplot(fig)
