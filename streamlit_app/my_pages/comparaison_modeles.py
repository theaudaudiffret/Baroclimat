import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show():
    st.title("🤖 Comparaison des modèles")
    st.write("Comparaison entre les différents modèles...")
    # -------------------------
    # Chargement des CSV
    # -------------------------
    st.header("Quel modèle choisir ?")

    # Chargement des fichiers pour l'analyse 4 bits
    df_lora_4_bits = pd.read_csv('Micro_category/Models_comparison/original_annotation_dataset/inference_lora_new/LORA_METRICS_NEW.csv')
    df_base_model_4_bits = pd.read_csv('Micro_category/Models_comparison/original_annotation_dataset/inference_base_model/METRICS_BASE_NEW.csv')

    # Chargement des fichiers pour l'analyse best/benchmark
    df_lora_best = pd.read_csv('Micro_category/Models_comparison/Final_model/resultats_lora.csv')
    df_base_best = pd.read_csv('Micro_category/Models_comparison/Final_model/resultats_base.csv')
    df_benchmark = pd.read_csv('Micro_category/Models_comparison/Final_model/Benchmark.csv')
    df_lora_4_bits = df_lora_4_bits.drop(columns=['Percentage of unsure'])
    df_base_model_4_bits = df_base_model_4_bits.drop(columns=['Percentage of unsure'])
    # Affichage des tableaux
    st.subheader("Metrics - Lora 4 Bits")
    st.dataframe(df_lora_4_bits)
    st.subheader("Metrics - Base Model 4 Bits")
    st.dataframe(df_base_model_4_bits)

    # -------------------------
    # Graphiques
    # -------------------------
    # Couleurs pastel à utiliser
    pastel_blue = "#AEC6CF"
    pastel_orange = "#FFB347"
    pastel_gray = "#CFCFC4"

    # 1. Analyse sur les CSV 4 bits
    st.subheader("Comparaison des métriques (4 bits)")
    metrics = ['Precision', 'Recall', 'F1_score', 'Top 1 Accuracy', 'Top 3 Accuracy']
    x = np.arange(len(metrics))  # positions des labels
    width = 0.35  # largeur des barres

    # Pour chaque modèle (chaque ligne) du CSV
    for model in df_lora_4_bits.index:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Graphique pour le modèle LoRA
        ax.bar(x - width/1.85, df_lora_4_bits.loc[model, metrics], width,
               label='Lora Model', color=pastel_blue)
        # Graphique pour le modèle de base
        ax.bar(x + width/1.85, df_base_model_4_bits.loc[model, metrics], width,
               label='Base Model', color=pastel_orange)
        
        # Configuration du graphique
        source = df_lora_4_bits.loc[model, "source"] if "source" in df_lora_4_bits.columns else f"Modèle {model}"
        ax.set_title(f'Comparaison des métriques pour {source}')
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Valeurs')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        plt.close(fig)

    # 2. Analyse sur les CSV best/benchmark
    st.header("Métriques - meilleurs modèles: ")
    st.subheader("Deepseek distill llama 8B et deepseek distill qwen 1B5")
    metrics = ['Precision', 'Recall', 'F1_score', 'Top 1 Accuracy', 'Top 3 Accuracy']
    x = np.arange(len(metrics))
    width = 0.20  # largeur différente pour trois barres

    for model in df_lora_best.index:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Barres pour le modèle LoRA
        ax.bar(x - width*1.05, df_lora_best.loc[model, metrics], width,
               label='Lora-Model', color=pastel_blue)
        # Barres pour le modèle de base
        ax.bar(x, df_base_best.loc[model, metrics], width,
               label='Base-Model', color=pastel_orange)
        # Barres pour le benchmark
        ax.bar(x + width*1.05, df_benchmark.loc[model, metrics], width,
               label='Benchmark semi-supervisé', color=pastel_gray)
        
        source = df_lora_best.loc[model, "source"] if "source" in df_lora_best.columns else f"Modèle {model}"
        ax.set_title(f'Comparaison des métriques pour {source}')
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Valeurs')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig)
