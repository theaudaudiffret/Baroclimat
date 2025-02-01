import torch
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import transformers



os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.cuda.is_available()








classes = [
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
    "Solutions innovantes"
    "Comportement de consommation",
    "Reforestation",

]

ds = load_dataset("csv", data_files="annotation_sous_thematiques.csv")
ds = ds['train'].train_test_split(test_size=0.2, shuffle = True)

ds['train'] = ds['train'].remove_columns(['title', 'date', 'order', 'presenter', 'editor', 'url', 'urlTvNews', 'containsWordGlobalWarming', 'media', 'month', 'day', 'label', 'Commentaires', 'nb_label'])
ds['test'] = ds['test'].remove_columns(['title', 'date', 'order', 'presenter', 'editor', 'url', 'urlTvNews', 'containsWordGlobalWarming', 'media', 'month', 'day', 'label', 'Commentaires', 'nb_label'])


def classification_up_to_3(text, categories, label_1, label_2, label_3,retry_attempts=5, base_delay=10):
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
    try:
        # Préparer le prompt pour le modèle
        prompt_template = f"""
        Donne seulement les 3 classes, parmi {categories}, qui correspondent le mieux au texte donné, de la plus probable à la moins probable.

        ## Règles :
        - Réponds uniquement avec un JSON strict du format suivant : {{ "Classe_1": "categorie_1", "Classe_2": "categorie_2", "Classe_2": "categorie_3" }}
        - Les catégories doivent être dans {categories}.
        - Les classes sont ordonnées de la plus probable à la moins probable.

        Texte : "{text}"

        Réponse :
        {{'Classe_1': {label_1},
        'Classe_2': {label_2},
        'Classe_3': {label_3}}}
        """
        return prompt_template
    except Exception as e:
        print(f"Outer exception: {e}")
        return None
    

os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':


    print(ds)