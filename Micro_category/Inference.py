import zipfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

import time
import json
import torch
import re
import ast
import pandas as pd


from sklearn.metrics import multilabel_confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


import subprocess
# Remplace "ton_token" par ton token Hugging Face
config_path = os.path.join("src", "config", "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)  # Charger le JSON correctement

# Extraire le token Hugging Face
token = config.get("huggingface", {}).get("api_token", "").strip()

if not token:
    raise ValueError("Token non défini dans le fichier config.json")

print("Token récupéré avec succès.")

# Exécute la commande Hugging Face CLI login
subprocess.run(["huggingface-cli", "login", "--token", token], check=True)

def compute_metrics(lora_paths, save_paths, path_data, full):
    if len(lora_paths) != len(save_paths):
        raise ValueError("Les listes lora_paths et save_paths doivent avoir la même longueur.")

    for j, lora_path in enumerate(lora_paths):

        # Charger la configuration du modèle LoRA
        config = PeftConfig.from_pretrained(lora_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        qa_model = PeftModel.from_pretrained(model, lora_path, offload_dir="offload_dir")

        categories = [
            "Gaz à effet de serre",
            "Elevage et utilisation des terres",
            "Pêche et chasse intensives",
            "Pollution plastique",
            "Déforestation",
            "Surconsommation",
            "Catastrophes naturelles",
            "Réchauffement climatique/canicules",
            "Sécheresse",
            "Couche d'ozone",
            "Feu de forêt",
            "Tensions alimentaires/famines",
            "Perte eau douce",
            "Hausse des océans et fonte des glaces",
            "Conséquence sociale",
            "Acidification des océans",
            "Biodiversité",
            "Pollution",
            "Energie renouvelable et nucléaire",
            "Transports décarbonés",
            "Engagements politiques et entreprises",
            "Activisme écologique",
            "Solutions innovantes",
            "Comportement de consommation",
            "Reforestation",
        ]

        def classification_up_to_k(text, tokenizer, qa_model, categories, k=3, retry_attempts=3, base_delay=10):
            """
            Classifie un texte dans une liste de k catégories avec un modèle Hugging Face (comme Falcon ou Llama).

            Arguments :
            - text : str, le texte à classifier.
            - tokenizer : tokenizer Hugging Face.
            - qa_model : modèle Hugging Face.
            - categories : list, liste des catégories possibles.
            - k : int, nombre maximum de catégories à retourner.
            - retry_attempts : int, nombre maximum de tentatives en cas d'erreur.
            - base_delay : int, temps d'attente initial (secondes) avant de retenter en cas d'erreur 429.

            Retourne :
            - dict : Un dictionnaire avec les k meilleures catégories et leurs scores.
            """
            # Préparer le prompt pour le modèle
            prompt_message = f"""
            Donne au maximum les {k} classes, parmi {categories}, qui correspondent le mieux au texte donné, de la plus probable à la moins probable.

            ## Règles :
            - Les catégories doivent être dans {categories}.
            - Les classes sont ordonnées de la plus probable à la moins probable.
            - Réponds uniquement avec un JSON strict du format suivant : {{ "Classe_1": "categorie_1", "Classe_2": "categorie_2", "Classe_3": "categorie_3" }}


            Texte : "{text}"

            Réponse :
            """

            # Tokeniser l'entrée
            batch = tokenizer(prompt_message, return_tensors='pt')

            # Envoyer le batch sur le bon device
            device = qa_model.device
            batch = {k: v.to(device) for k, v in batch.items()}

            for attempt in range(retry_attempts):
                    # Effectuer l'inférence avec le modèle
                with torch.cuda.amp.autocast():
                        output_tokens = qa_model.generate(**batch, max_new_tokens=200)

                # Décoder la réponse
                response = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
                # Extraire uniquement la partie après "Réponse :"
                response = response.split("Réponse :\n")[-1].strip()

                # Recherche du dictionnaire entre accolades
                #match = re.search(r'\{.*\}', response)
                match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

                if match:
                    # # Conversion de la chaîne en dictionnaire
                    # dict_str = match.group(0)
                    # #result_dict = ast.literal_eval(dict_str)
                    # result_dict = json.loads(dict_str)

                    # return result_dict
                    dict_str = match.group(0)
                    try:
                        result_dict = json.loads(dict_str)
                        return result_dict
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        print(f"Invalid JSON response: {dict_str}")
                        continue  # Retry if JSON is invalid
            


            return(None)


        path = path_data
        df = pd.read_csv(path, sep=";")
        df['label'] = 1
        if full == False:
            df = df.tail(int(0.2 * len(df)))
            
        df["prediction_label"] = None
        for idx, row in df.iterrows():
            df.at[idx, "prediction_label"] = classification_up_to_k(row["description"], tokenizer, qa_model, categories, k=3)
            time.sleep(1)  # Attendre 4 secondes entre chaque requête
        def safe_extract(x, key):
            """
            Fonction qui extrait une valeur de 'x' en s'assurant que :
            - 'x' est bien un dictionnaire
            - 'key' existe dans 'x'
            - La valeur associée à 'key' est une liste avec au moins 'index+1' éléments
            """
            try:
                # Si x est un dictionnaire et contient la clé, extraire l'index demandé
                if isinstance(x, dict):
                    return x.get(key, [None, None])  # Valeur par défaut = None (pour NaN)
            except (TypeError, IndexError):
                return None  # Retourne None pour faciliter la suppression des NaN

        # Extraction des labels 
        for i in range(1, 4):
            df[f"prediction_label_{i}"] = df["prediction_label"].apply(lambda x: safe_extract(x, f"Classe_{i}"))

        # Suppression des lignes avec NaN dans les colonnes créées
        df = df.dropna(subset=[f"prediction_label_{i}" for i in range(1, 4)])


        df.to_csv(save_paths[j], index=False)


    #Mapping pour faire coïncider les noms codés des labels avec les prédictions OpenAI
    mapping = {
        "Gaz à effet de serre": "gaz_effet_de_serre",
        "Elevage et utilisation des terres": "agriculture_et_utilisation_du_sol",
        "Pêche et chasse intensives": "peche_et_chasse",
        "Pollution plastique": "pollution",
        "Déforestation": "deforestation",
        "Surconsommation": "surconsommation",
        "Catastrophes naturelles": "catastrophes_naturelles",
        "Réchauffement climatique/canicules": "rechauffement_climatique_canicule",
        "Sécheresse": "secheresse",
        "Couche d'ozone": "couche_ozone",
        "Feu de forêt": "feu_foret",
        "Tensions alimentaires/famines": "tension_alim_famines",
        "Perte eau douce": "eau_potable",
        "Hausse des océans et fonte des glaces": "hausse_niveau_mer_fonte_glace",
        "Conséquence sociale": "consequence_sociale",
        "Acidification des océans": "acidification_ocean",
        "Biodiversité": "perte_biodiversite",
        "Pollution": "pollution",
        "Energie renouvelable et nucléaire": "energies_renouvelables_et_nucleaires",
        "Transports décarbonés": "transport_decarbone",
        "Engagements politiques et entreprises": "engagement_politique_et_entreprises",
        "Activisme écologique": "activisme_eco",
        "Solutions innovantes": "solution_innovante",
        "Comportement de consommation": "comportement_consommateur",
        "Reforestation": "reforestation"
    }


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


    def format_labels(labels, mapping):
        """
        Formate les labels en utilisant le mapping, mais conserve les labels déjà formatés.

        Arguments :
        - labels : list, la liste des labels à formater.
        - mapping : dict, le dictionnaire de correspondance des labels.

        Retourne :
        - list, la liste des labels formatés.
        """
        formatted_labels = []
        for label in labels:
            # Si le label est déjà dans le bon format (valeur du mapping), on le conserve
            if label in mapping.values():
                formatted_labels.append(label)
            # Sinon, on utilise le mapping pour le formater
            else:
                formatted_label = mapping.get(label, "N/A")
                if formatted_label != "N/A":
                    formatted_labels.append(formatted_label)
        return formatted_labels

    def get_perf(df_results, pred_col):
        y_true = df_results["label"]
        y_pred = df_results[pred_col]

        # Combine all thematic keys for MultiLabelBinarizer
        full_dic = {**cause_thematiques, **consequence_thematiques, **solution_thematiques}
        mlb = MultiLabelBinarizer(classes=list(full_dic.keys()))

        # Transform true and predicted labels
        y_true_binarized = mlb.fit_transform(y_true)
        y_pred_binarized = mlb.transform(y_pred)

        # Calculate metrics with zero_division=0 to avoid warnings
        precision = precision_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0) * 100
        recall = recall_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0) * 100

        # Top-1 and Top-3 accuracy
        def top1_accuracy(preds, labels):
            if len(preds) == 0:
                return 0
            return 1 if preds[0] in labels else 0

        def top3_accuracy(preds, labels):
            return 1 if any(pred in labels for pred in preds[:3]) else 0

        top1_acc = round(df_results.apply(lambda x: top1_accuracy(x[pred_col], x["label"]), axis=1).sum() / len(df_results) * 100, 2)
        top3_acc = round(df_results.apply(lambda x: top3_accuracy(x[pred_col], x["label"]), axis=1).sum() / len(df_results) * 100, 2)

        # Percentage of unsure predictions (empty predictions)
        unsure_percentage = round(df_results.apply(lambda x: 1 if len(x[pred_col]) == 0 else 0, axis=1).sum() / len(df_results) * 100, 2)

        # Create a DataFrame with the results
        df_score_global = pd.DataFrame({
            "Precision": [precision],
            "Recall": [recall],
            "F1_score": [f1],
            "Top 1 Accuracy": [top1_acc],
            "Top 3 Accuracy": [top3_acc],
            "Percentage of unsure": [unsure_percentage]
        })

        return df_score_global.round(2)


    def pipe(path):
        df = pd.read_csv(path)

        # Create a copy to avoid SettingWithCopyWarning
        df_results = df[
            ["description", "label_1", "label_2", "label_3",
            "prediction_label_1", "prediction_label_2", "prediction_label_3"]
        ].copy()

        # Format true labels
        df_results["label"] = df_results[["label_1", "label_2", "label_3"]].apply(
            lambda x: format_labels(x.dropna().tolist(), mapping), axis=1
        )
        # Format predicted labels
        df_results["pred_label"] = df_results[
            ["prediction_label_1", "prediction_label_2", "prediction_label_3"]
        ].apply(
            lambda x: format_labels(x.dropna().tolist(), mapping), axis=1
        )
        # Affiche les labels bruts avant de les passer à format_labels


        # Count hallucinations (N/A in predictions)
        counter = sum(1 for preds in df_results["pred_label"] if "N/A" in preds)
        print(f"Nombre de documents avec une hallucination: {counter}")

        return get_perf(df_results, "pred_label")
    result = pd.DataFrame()

    for path in save_paths:
        df_result = pipe(path)

        if df_result is not None and not df_result.empty:
            filename = os.path.basename(path)
            index_name = filename.replace(".csv", "").split("_")[-1]

            df_result["source"] = index_name
            result = pd.concat([result, df_result], ignore_index=True)

    result.set_index("source", inplace=True)


    file_path = "Micro_category/Metrics/results.csv"
    write_header = not os.path.exists(file_path)  # Écrire l'en-tête seulement si le fichier n'existe pas

    result.to_csv(file_path, mode="a", header=write_header)

    return result

if __name__ == "__main__":
    compute_metrics(lora_paths=["models/lora-distill-llama-8b-boost"],
                    save_paths=["Micro_category/Inference/predictions_Micro-lora-8B-2.csv"],
                    path_data="data/Annotations_macro_thematiques_new.csv",
                    full=False
                    )
    # Put true as an argument for full if you want to make the inference on the whole dataset
    # False is for splitting the dataset between train and test 
    
