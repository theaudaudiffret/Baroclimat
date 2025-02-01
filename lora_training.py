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



model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Use float16 for faster computation
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



config = LoraConfig(
    r=6,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)



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
    
mapped_ds_dataset = ds.map(lambda samples: tokenizer(classification_up_to_3(samples['description'], classes, samples['label_1'], samples['label_2'], samples['label_3'])))

os.environ["WANDB_DISABLED"] = "true"


trainer = transformers.Trainer(
    model=model,
    train_dataset=mapped_ds_dataset["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        max_steps=2,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=1,
        output_dir='outputs',
        report_to="none"  # Désactive les rapports vers W&B

    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if __name__ == '__main__':
    print_trainable_parameters(model)
    trainer.train()


    # Save the model
    local_model_path = "./my_local_model"
    # Sauvegarde en local
    model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    print("Model saved")