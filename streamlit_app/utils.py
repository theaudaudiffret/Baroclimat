import altair as alt
import matplotlib as plt
import numpy as np
import pandas as pd

def climate_topic_dashboard(df):
    """
    Creates a Streamlit dashboard for the Climate Topic dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the Climate Topic dataset.

    Returns
    -------
    alt.Chart
        The generated Streamlit dashboard as an Altair Chart object.
    """

    media_colors = {
        "arte": "#f08400",
        "fr3-idf": "#8fc7ff",
        "france2": "#ed2f28",
        "france5": "#f4a8a7",
        "itele": "#50ad9a",
        "m6": "#90ee9d",
        "tf1": "#3266c8",
        "France2": "#ed2f28",
        "France3": "#8fc7ff",
        "France5": "#f4a8a7",
        "M6": "#90ee9d",
        "Arte": "#f08400",
        "TF1": "#3266c8",
    }

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["number_articles"] = df.groupby("date")["date"].transform("count")
    df["number_climate"] = df.groupby("date")["label"].transform("sum")

    # Calculate average proportion on a rolling window of climate articles by media and date
    window_size = 60
    df_groupby = df.copy()
    avg, std = (
        df_groupby[df_groupby["label"] == 1]["confidence"].mean(),
        df_groupby[df_groupby["label"] == 1]["confidence"].std(),
    )
    df_groupby["avg_proba"] = df_groupby[df_groupby["label"] == 1]["confidence"].rolling(10).mean()
    df_groupby["std_proba"] = df_groupby[df["label"] == 1]["confidence"].rolling(10).std()
    # df_groupby["climate_related_selected"] = np.where(
    #     (df_groupby["confidence"] > avg - std) & (df_groupby["label"] == 1), 1, 0
    # )
    df_groupby["climate_related_selected"] = np.where((df_groupby["label"] == 1), 1, 0
    )
    df_groupby = (
        df_groupby.groupby(["media", "date"])[["label", "climate_related_selected"]].mean().reset_index()
    )
    df_groupby["Proportion"] = df_groupby["label"].transform(
        lambda x: x.rolling(window_size, min_periods=1).mean()
    )
    df_groupby["Proportion_selected"] = df_groupby["climate_related_selected"].transform(
        lambda x: x.rolling(window_size, min_periods=1).mean()
    )

    # # Confidence interval bounds
    # z_alpha = 1.96
    # df_groupby["number_climate"] = (
    #     df.groupby(["media", "date"])["label"].sum().rolling(window_size, min_periods=1).sum().values
    # )
    # df_groupby["number_articles"] = (
    #     df.groupby(["media", "date"])["label"].count().rolling(window_size, min_periods=1).sum().values
    # )
    # df_groupby["proportion_tilde"] = (df_groupby["number_climate"] + 2) / (df_groupby["number_articles"] + 4)
    # df_groupby["confidence_low"] = df_groupby["Proportion_selected"] - z_alpha * np.sqrt(
    #     df_groupby["Proportion_selected"]
    #     * (1 - df_groupby["Proportion_selected"])
    #     / (df_groupby["number_articles"] + 4)
    # )
    # df_groupby["confidence_high"] = df_groupby["Proportion_selected"] + z_alpha * np.sqrt(
    #     df_groupby["Proportion_selected"]
    #     * (1 - df_groupby["Proportion_selected"])
    #     / (df_groupby["number_articles"] + 4)
    # )

    media_selected = df_groupby["media"].unique()
    media_colors = {media: media_colors[media] for media in media_selected}

    chart = (
        alt.Chart(df_groupby)
        .mark_line()
        .encode(
            x=alt.X("date:T", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("Proportion_selected:Q", axis=alt.Axis(format="%")),
            color=alt.Color(
                "media:N", scale=alt.Scale(domain=list(media_colors.keys()), range=list(media_colors.values()))
            ),
            tooltip=["date:T", "Proportion_selected:Q", "media:N"],
        )
        .properties(
            width=600,
            height=400,
            title="Evolution de la proportion des articles climatiques",
        )
    )

    
    return chart
