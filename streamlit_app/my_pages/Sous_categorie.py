import streamlit as st
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


def show():
    st.title("📊 De quel sujet a-t-on parlé sur TF1 et France 2 en 2024?")
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
        csv_path = "Micro_category/Inference/predictions_2024_Micro-lora-8B-new.csv"
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

        cause_thematiques = {"gaz_effet_de_serre": "gaz_effet_de_serre",
                            "agriculture_et_utilisation_du_sol": "agriculture_et_utilisation_du_sol",
                            "peche_et_chasse": "peche_et_chasse",
                            "intrants_chimique_pollution_plastique": "intrants_chimique_pollution_plastique",
                            "surconsommation": "surconsommation",
                            "deforestation": "deforestation"}



        consequence_thematiques = {"catastrophes_naturelles":"catastrophes_naturelles",
                                "rechauffement_climatique_canicule":"rechauffement_climatique_canicule",
                                    "secheresse":"secheresse",
                                    "couche_ozone":"couche_ozone",
                                    "feu_foret":"feu_foret",
                                    "tension_alim_famines":"tension_alim_famines",
                                    "eau_potable":"eau_potable",
                                    "hausse_niveau_mer_fonte_glace":"hausse_niveau_mer_fonte_glace",
                                    "consequence_sociale":"consequence_sociale",
                                    "acidification_ocean":"acidification_ocean",
                                    "perte_biodiversite":"perte_biodiversite",
                                    "pollution":"pollution"}

        solution_thematiques = {"energies_renouvelables_et_nucleaires": "energies_renouvelables_et_nucleaires",
                                "transport_decarbone": "transport_decarbone",
                                "engagement_politique_et_entreprises": "engagement_politique_et_entreprises",
                                "activisme_eco": "activisme_eco",
                                "solution_innovante": "solution_innovante",
                                "comportement_consommateur": "comportement_consommateur",
                                "reforestation": "reforestation"}






    elif st.session_state.selection == "Macro":
        csv_path = "Macro_category/Inference/predictions_2024_Macro-lora-8B-1-cat.csv"
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
        df["date"] = df["date"].str.split(" ").str[0]

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
            "confidence", "prediction_label", "authors", "editor"
        ]
        df_filtre.drop(columns=[col for col in colonnes_a_supprimer if col in df_filtre.columns], inplace=True)

        # ======================================================
        # 3. Filtre par catégorie (présente dans l'une des colonnes de prédiction)
        # ======================================================
        if st.session_state.selection == "Micro":
            cat_cols = ["prediction_label_1", "prediction_label_2", "prediction_label_3"]
        else:  # Macro
            cat_cols = ["prediction_label_1"]
            # On peut également drop les colonnes inutiles si elles existent
            df_filtre.drop(columns=[col for col in ["prediction_label_2", "prediction_label_3"] if col in df_filtre.columns], inplace=True)

        all_cats = categories.copy()
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
        if df_filtre.empty:
            st.write("Aucune donnée pour la catégorie sélectionnée.")
        else:
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
            if st.session_state.selection == "Micro":
                # Concaténation des 3 colonnes pour Micro
                all_preds = pd.concat([
                    df_filtre["prediction_label_1"],
                    df_filtre["prediction_label_2"],
                    df_filtre["prediction_label_3"]
                ])
            else:
                # Pour Macro, seule la colonne prediction_label_1 est utilisée
                all_preds = df_filtre["prediction_label_1"]
                
            all_preds = all_preds[all_preds.isin(categories)]
            #cat_counts = all_preds.value_counts().sort_values(ascending=False)
            cat_counts = all_preds.value_counts().reindex(sorted(categories), fill_value=0).sort_values(ascending=False)

            pastel_palette = sns.color_palette("pastel", len(cat_counts))

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax1, palette=pastel_palette)
                ax1.set_xlabel("Catégorie")
                ax1.set_ylabel("Occurrences")
                ax1.set_title("Occurrences par catégorie")
                ax1.tick_params(axis='x', rotation=90)
                st.pyplot(fig1)

            with col2:
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


            # Fusionner les trois colonnes de prédiction
            df_long = pd.melt(
                df_filtre,
                id_vars=["date"],
                value_vars=["prediction_label_1", "prediction_label_2", "prediction_label_3"],
                var_name="prediction_rank",
                value_name="category"
            )

            # Garder uniquement les catégories à visualiser
            df_long = df_long[df_long["category"].isin(categories)]

            # Extraire l'année et le mois comme nouvelle colonne "month"
            df_long["month"] = df_long["date"].dt.to_period("M").dt.to_timestamp()

            # Compter les occurrences par mois et catégorie
            df_counts = (
                df_long.groupby(["month", "category"])
                .size()
                .reset_index(name="count")
            )

            # Calculer le total mensuel pour chaque mois
            total_per_month = df_counts.groupby("month")["count"].sum().reset_index(name="total")
            df_counts = df_counts.merge(total_per_month, on="month")
            df_counts["share"] = df_counts["count"] / df_counts["total"] * 100

            # Créer le graphique en aires empilées (stacked area)
            area_chart = alt.Chart(df_counts).mark_area().encode(
                x=alt.X("month:T", title="Mois"),
                y=alt.Y("share:Q", stack="normalize", title="Part (%)"),
                color=alt.Color("category:N", title="Catégorie"),
                tooltip=["month:T", "category:N", alt.Tooltip("share:Q", format=".2f")]
            ).properties(
                width=800,
                height=400,
                title="Part moyenne mensuelle des sous-catégories"
            ).interactive()

            st.altair_chart(area_chart, use_container_width=True)


            if st.session_state.selection == "Micro":




                # Convertir la date si ce n'est pas déjà fait
                df_filtre["date"] = pd.to_datetime(df_filtre["date"])

                # Extraire le mois sous forme "YYYY-MM"
                df_filtre["mois"] = df_filtre["date"].dt.to_period("M").astype(str)

                # Fonction qui renvoie les thématiques d'un article (sans doublon)
                def get_thématiques_article(row):
                    labels = [row["prediction_label_1"], row["prediction_label_2"], row["prediction_label_3"]]
                    thématiques = set()
                    for label in labels:
                        if label in cause_thematiques:
                            thématiques.add("cause")
                        elif label in consequence_thematiques:
                            thématiques.add("conséquence")
                        elif label in solution_thematiques:
                            thématiques.add("solution")
                    return list(thématiques)

                # Appliquer la fonction
                df_filtre["thématiques_par_article"] = df_filtre.apply(get_thématiques_article, axis=1)

                # Exploser pour avoir une ligne par thématique
                df_exploded = df_filtre.explode("thématiques_par_article")

                # Filtrer avec le multiselect
                thematiques_options = ["cause", "conséquence", "solution"]
                selected_thematiques = st.multiselect(
                    "Sélectionnez les types de thématiques à visualiser",
                    options=thematiques_options,
                    default=thematiques_options
                )

                df_exploded = df_exploded[df_exploded["thématiques_par_article"].isin(selected_thematiques)]

                # Grouper par mois et thématique
                df_count = df_exploded.groupby(["mois", "thématiques_par_article"]).size().reset_index(name="Occurrences")

                # Tracer le graphique
                chart = alt.Chart(df_count).mark_bar().encode(
                    x=alt.X("mois:N", title="Mois"),
                    y=alt.Y("Occurrences:Q", title="Nombre d'articles"),
                    color=alt.Color("thématiques_par_article:N", title="Thématique"),
                    tooltip=["mois", "thématiques_par_article", "Occurrences"]
                ).properties(
                    width=700,
                    height=400,
                    title="Nombre d’articles par mois et par thématique"
                )

                st.altair_chart(chart, use_container_width=True)