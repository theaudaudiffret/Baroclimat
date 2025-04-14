import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

def show():
       st.title("🤖 Comparaison des modèles.")
       st.write("Les modèles ont été comparé sur la base de données de l'annotation originale faite par les étudiants de l'année passée. "
              "Nous comparons dans un premier temps les performances des modèles entre eux et avec leur version fine-tunée. "
              "Le but est ensuite de garder le meilleur modèle et de comparer les résultats avec ceux obtenus par le modèle de clustering semi-supervisé qui avait été mis en place et qui utilise FastText.")
       # -------------------------
       # Chargement des CSV
       # -------------------------
       st.header("Choix du modèle")
       st.write("Nous avons essayé de fine-tuner plusieurs modèles en open-source sur Hugging Face pour voir si nous pouvions améliorer les perfomances de ces petits LLMs.\n"
                "Pour des raisons de capacités et de coût énergétique nous avons fait la pré-sélection sur des modèles chargés en '4-bit quantization'\n")


       # Chargement des fichiers pour l'analyse 4 bits
       df_lora_4_bits = pd.read_csv('Micro_category/Models_comparison/original_annotation_dataset/inference_lora_new/LORA_METRICS_NEW.csv')
       df_base_model_4_bits = pd.read_csv('Micro_category/Models_comparison/original_annotation_dataset/inference_base_model/METRICS_BASE_NEW.csv')

       # Chargement des fichiers pour l'analyse best/benchmark
       df_lora_best = pd.read_csv('Micro_category/Models_comparison/Final_model/resultats_lora.csv')
       df_base_best = pd.read_csv('Micro_category/Models_comparison/Final_model/resultats_base.csv')
       df_benchmark = pd.read_csv('Micro_category/Models_comparison/Final_model/Benchmark.csv')
       df_gpt = pd.read_csv('Micro_category/Models_comparison/Final_model/ChatGPT.csv')
       df_lora_4_bits = df_lora_4_bits.drop(columns=['Percentage of unsure'])
       df_base_model_4_bits = df_base_model_4_bits.drop(columns=['Percentage of unsure'])

       # Affichage côte à côte
       col1, col2 = st.columns(2)

       with col1:
              st.subheader("Metrics - Lora 4 Bits")
              st.dataframe(df_lora_4_bits)

       with col2:
              st.subheader("Metrics - Base Model 4 Bits")
              st.dataframe(df_base_model_4_bits)
       # -------------------------
       # Graphiques
       # -------------------------
       # Couleurs pastel à utiliser
       pastel_blue = "#AEC6CF"
       pastel_orange = "#FFB347"
       pastel_gray = "#CFCFC4"
       pastel_light_pink = "#FFC1C1"


       # 1. Analyse sur les CSV 4 bits
       st.subheader("Comparaison des métriques (4 bits)")
       metrics = ['Precision', 'Recall', 'F1_score', 'Top 1 Accuracy', 'Top 3 Accuracy']
       x = np.arange(len(metrics))  # positions des labels
       width = 0.35  # largeur des barres

       cols = st.columns(2)  # deux graphiques par ligne
       for i, model in enumerate(df_lora_4_bits.index):
              col = cols[i % 2]  # alterne entre col1 et col2
              with col:
                     fig, ax = plt.subplots(figsize=(10, 6))
                     # Graphique pour le modèle LoRA
                     ax.bar(x - width/1.85, df_lora_4_bits.loc[model, metrics], width,
                            label='Lora Model', color=pastel_blue)
                     # Graphique pour le modèle de base
                     ax.bar(x + width/1.85, df_base_model_4_bits.loc[model, metrics], width,
                            label='Base Model', color=pastel_orange)
                     
                     source = df_lora_4_bits.loc[model, "source"] if "source" in df_lora_4_bits.columns else f"Modèle {model}"
                     ax.set_title(f'Comparaison des métriques pour {source}')
                     ax.set_xlabel('Métriques')
                     ax.set_ylabel('Valeurs')
                     ax.set_xticks(x)
                     ax.set_xticklabels(metrics)
                     ax.legend()
                     st.pyplot(fig)
                     plt.close(fig)

       # 2. Analyse sur les CSV best/benchmark
       st.header("Modèle retenus")
       st.write("Nous avons retenu le modèle Distill-Llama-8B qui a 8 milliards de apramètres en raison de ses performances ainsi que le modèle Distill-Qwen-1.5B pour étudier les résultats d'un modèle plus petit.\n "
                "Ces modèles ont été fine tunés avec LoRa et chargés en float16 ce qui les rend entraînables sur des gpu avec 24 Go de RAM. \n"
                "Nous avons comparé les résultats fine-tunés avec ceux des modèles de base ainsi qu'avec ceux obtenus par le modèle de FastText qui avait été mis en place et avec les résultats obtenus avec GPT 3.5 Turbo (175b de paramètres). \n")
       st.subheader("Deepseek distill llama 8B et deepseek distill qwen 1B5")
       metrics = ['Precision', 'Recall', 'F1_score', 'Top 1 Accuracy', 'Top 3 Accuracy']
       x = np.arange(len(metrics))
       width = 0.15

       cols = st.columns(2)  # deux graphiques par ligne
       for i, model in enumerate(df_lora_best.index):
              col = cols[i % 2]
              with col:
                     fig, ax = plt.subplots(figsize=(10, 6))
                     ax.bar(x - width*1.5, df_lora_best.loc[model, metrics], width,
                            label='Lora-Model', color=pastel_blue)
                     ax.bar(x - width*0.5, df_base_best.loc[model, metrics], width,
                            label='Base-Model', color=pastel_orange)
                     ax.bar(x + width*0.5, df_benchmark.loc[model, metrics], width,
                            label='Ancien model (FastText)', color=pastel_gray)
                     ax.bar(x + width*1.5, df_gpt.loc[model, metrics], width,
                            label='GPT-3.5', color=pastel_light_pink)

                     source = df_lora_best.loc[model, "source"] if "source" in df_lora_best.columns else f"Modèle {model}"
                     ax.set_title(f'Comparaison des métriques pour {source}')
                     ax.set_xlabel('Métriques')
                     ax.set_ylabel('Valeurs')
                     ax.set_xticks(x)
                     ax.set_xticklabels(metrics)
                     ax.legend()
                     st.pyplot(fig)
                     plt.close(fig)
       st.subheader("Score énergétique des modèles testés")

       data = {
       "Modèle": [
              "mistral large", "Deepseek-Qwen-7B", "small mistral", "mistral 8x7b",
              "ministral 8b", "mistral 7b", "llama 3B", "Deepseek-Qwen-1.5B",
              "falcon-7B", "chatgpt_ancien-3.5"
       ],
       "Taille": [123, 7, 24, 12.9, 8, 7, 3, 1.5, 7, 175],
       "Energy": [1100, 17.5, 1100, 615.39, 20.87, 19.01, 7.5, 7.5, 15.88, 1100],
       "Open Source": [False, True, False, True, True, True, True, True, True, False]
       }

       df = pd.DataFrame(data)
       df = df[df["Taille"].notnull()].copy()

       # Calque des points
       points = alt.Chart(df).mark_circle().encode(
       x=alt.X("Taille:Q", scale=alt.Scale(type='log'), title="Taille du modèle (log scale, en B)"),
       y=alt.Y("Energy:Q", scale=alt.Scale(type='log'), title="AI Energy Score (log scale, Wh pour 1000 inférences)"),
       color=alt.Color("Open Source:N",
                     scale=alt.Scale(domain=[True, False], range=["green", "red"]),
                     legend=alt.Legend(title="Open Source")),
       size=alt.Size("Taille:Q",
                     scale=alt.Scale(range=[50, 400]),
                     legend=alt.Legend(title="Taille du modèle (B)")),
       tooltip=["Modèle", "Taille", "Energy", "Open Source"]
       )

       # Calque des étiquettes (labels)
       labels = alt.Chart(df).mark_text(
       align='left',
       baseline='middle',
       dx=7,  # petit décalage en X pour éviter chevauchement
       fontSize=11
       ).encode(
       x=alt.X("Taille:Q", scale=alt.Scale(type='log')),
       y=alt.Y("Energy:Q", scale=alt.Scale(type='log')),
       text="Modèle"
       )

       # Combinaison des deux calques
       chart = (points + labels).properties(
       title="Modèles IA : Taille vs Énergie (log-log, avec étiquettes visibles)",
       width=850,
       height=450
       ).interactive()

       st.altair_chart(chart)
