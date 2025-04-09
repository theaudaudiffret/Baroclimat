import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import climate_topic_dashboard
import datetime
import altair as alt

def show():
    st.title("📊 Dashboard 2024")
    st.write("Bienvenue sur la page du Dashboard.")
    st.write("Cette page vous permet de visualiser les résultats de l'inférence sur le modèle 2024.")
    st.write("Vous pouvez explorer les pourcentages d'articles catégorisés 'Climat' par période (jour, semaine, mois) ainsi que filtrer les articles par date et média.")


    # 📥 Lecture du fichier
    try:
        df = pd.read_csv("data/2024_JT_TF1_F2.csv")
    except FileNotFoundError:
        st.error("Fichier introuvable.")
        return

    # ⚠️ Nettoyage des dates AVANT TOUT
    if 'date' not in df.columns:
        st.error("La colonne 'date' est manquante.")
        return

    df["date"] = pd.to_datetime(df["date"], format='mixed', errors="coerce")  # mixed pour gerer le format

    # Vérification colonne 'label'
    if 'label' not in df.columns:
        st.error("Le fichier doit contenir une colonne 'label'.")
        return

    st.subheader("🔍 Vision Globale")
    #Sélection de l'échelle de temps
    st.write("Cette section vous permet d'observer la tendance des articles catégorisés 'Climat' sur différentes périodes : jour, semaine ou mois.")
    time_scale = st.selectbox("Choisissez l'échelle de temps", ['Jour', 'Semaine', 'Mois'])

    # Agrégation selon l'échelle
    if time_scale == 'Jour':
        df_grouped = df.groupby(df['date'].dt.date).agg(
            total_articles=('date', 'size'),
            articles_climat=('label', lambda x: (x == 1).sum())
        )
        df_grouped.index = pd.to_datetime(df_grouped.index)
        title = "📊 Pourcentage des articles catégorisés Climat par jour"

    elif time_scale == 'Semaine':
        df_grouped = df.groupby(df['date'].dt.to_period('W')).agg(
            total_articles=('date', 'size'),
            articles_climat=('label', lambda x: (x == 1).sum())
        )
        df_grouped.index = df_grouped.index.to_timestamp()  # <- pour convertir en datetime
        title = "📊 Pourcentage des articles catégorisés Climat par semaine"

    elif time_scale == 'Mois':
        df_grouped = df.groupby(df['date'].dt.to_period('M')).agg(
            total_articles=('date', 'size'),
            articles_climat=('label', lambda x: (x == 1).sum())
        )
        df_grouped.index = df_grouped.index.to_timestamp()
        title = "📊 Pourcentage des articles catégorisés Climat par mois"

    # Calcul du ratio climat
    df_grouped['ratio'] = (df_grouped['articles_climat'] / df_grouped['total_articles']) * 100



    #sur le modèle des anciens
    df_grouped.index = df_grouped.index.astype("datetime64[ns]")
    df_grouped = df_grouped.reset_index()
    df_grouped.rename(columns={"index": "date"}, inplace=True)


    chart = (
        alt.Chart(df_grouped)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("ratio:Q", title="Ratio d'articles climat (%)"),
            tooltip=["date:T", "ratio:Q"]
        )
    ).properties(
        width=700,
        height=400,
        title="Évolution du ratio d'articles climat"
    )

    st.altair_chart(chart, use_container_width=True)

    chart_histo = (
        alt.Chart(df_grouped)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date",axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("ratio:Q", title="Ratio d'articles climat (%)"),
            tooltip=["date:T", "ratio:Q"],
            color=alt.value("#4C78A8")  # couleur personnalisable
        )
        .properties(
            width=700,
            height=400,
            title="Histogramme du ratio d'articles climat par date"
        )
    )

    st.altair_chart(chart_histo, use_container_width=True)
    
    # climate_chart = climate_topic_dashboard_all(df_grouped).interactive()
    # st.altair_chart(climate_chart, use_container_width=True)

    #affichage type ancien 

    df["year"] = df["date"].dt.year


    # Global figures
    col = st.columns((1.5, 1.5, 1.5), gap="medium")
    today = df["date"].max()

    st.markdown(
        """
    <style>
    .big-font {
        font-size:45px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


    with col[0]:
        with st.container():
            st.subheader("Nombre total de reportages TV (TF1 et France 2) diffusés sur l'année")
            df_week = df[df["date"] > today - datetime.timedelta(days=7)]
            st.markdown(f"""<p class="big-font">{df.shape[0]}</p>""", unsafe_allow_html=True)

    with col[1]:
        with st.container():
            st.subheader("Pourcentage d'articles catégorisés climat sur l'ensemble de l'année")
            percentage_climate = df["label"].mean() * 100
            if percentage_climate > 10:  # percentage to diplay in green if above 10%
                st.markdown(
                    f"""<p class="big-font" style="color: #228B22;">{percentage_climate:.2f} %</p>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<p class="big-font" style="color: #ff0000;">{percentage_climate:.2f} %</p>""",
                    unsafe_allow_html=True,
                )

    with col[2]:
        with st.container():
            st.subheader("Evolution de la couverture médiatique du climat sur le dernier mois")
            df_month = df[df["date"] > today - datetime.timedelta(days=30)]
            evolution = (df_week["label"].mean() - df_month["label"].mean()) * 100
            if evolution > 0:
                st.markdown(
                    f"""<p class="big-font" style="color: #228B22;">{evolution:.2f} %</p>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<p class="big-font" style="color: #ff0000;">{evolution:.2f} %</p>""",
                    unsafe_allow_html=True,
                )


    # climate_chart = climate_topic_dashboard(df).interactive()
    # st.altair_chart(climate_chart, use_container_width=True)



    st.subheader("🔎 Vision Précise")
    st.write("Cette section vous permet de filtrer les articles par date de début, date de fin, et média. Vous pouvez également visualiser les articles spécifiques à certaines catégories ou thématiques.")
    
    # Filtres personnalisés
    col_filters = st.columns((1, 1, 1), gap="small")

    with col_filters[0]:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        start_date = st.date_input("Date de début", min_date, min_value=min_date, max_value=max_date)

    with col_filters[1]:
        end_date = st.date_input("Date de fin", max_date, min_value=min_date, max_value=max_date)

    with col_filters[2]:
        selected_media = st.selectbox("Sélectionnez un média", ["Tous"] + sorted(df["media"].unique()))

    if start_date > end_date:
        st.error("Erreur : la date de fin doit être postérieure à la date de début.")
        return

    # Filter data based on selected filters
    df_filtered = df.copy()
    df_filtered = df_filtered[
        (df_filtered["date"] >= np.datetime64(start_date)) & (df_filtered["date"] <= np.datetime64(end_date))
    ]
    if selected_media != "Tous":
        df_filtered = df_filtered[df_filtered["media"] == selected_media]

    # Display the total percentage of climate scripts over the selected period
    st.subheader("Couverture médiatique du climat sur la période sélectionnée")
    percentage_climate_filtered = df_filtered["label"].mean() * 100
    if percentage_climate_filtered > 10:  # percentage to diplay in green if above 10%
        st.markdown(
            f"""<p class="big-font" style="color: #228B22;">{percentage_climate_filtered:.2f} %</p>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<p class="big-font" style="color: #ff0000;">{percentage_climate_filtered:.2f} %</p>""",
            unsafe_allow_html=True,
        )

    climate_chart_filtered = climate_topic_dashboard(df_filtered).interactive()
    st.altair_chart(climate_chart_filtered, use_container_width=True)




