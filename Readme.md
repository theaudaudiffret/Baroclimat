# Repository Organization

```
📂 data/                            <-- Labeled data used for training and computing metrics
📂 Macro_category/                  <-- Subcategory segmentation using 5 macro categories
│  │── 📂 Inference/                <-- CSV containing the results of the inference: text + categories
│  │── 📂 Metrics/                  <-- CSV containing the metrics for the model
│  │── 📄 Inference.py              <-- Run this to make an inference
│  │── 📄 lora_training_MACRO.py    <-- Run this to fine-tune a model on the macro categories
│
📂 Micro_category/                  <-- Subcategory segmentation using 25 macro categories
│  │── 📂 Inference/                <-- CSV containing the results of the inference: text + categories
│  │── 📂 Metrics/                  <-- CSV containing the metrics for the model
│  │── 📂 Models_comparison/        <-- Comparison of fine-tuning different models
│  │   │── 📂 Final_model/
│  │   │   │── 📂 inference_final_model/          <-- Inference of the 2 best models
│  │   │   │── 📂 lora_distill_llama_8b_boost2/   <-- Inference of the 2 best models
│  │   │   │── 📂 lora_distill_Qwen_1.5B_boost/
│  │   │   │── 📄 Benchmark.csv
│  │   │   │── 📄 resultats_base.csv
│  │   │   │── 📄 resultats_lora.csv
│  │   │── 📂 original_annotation_dataset/         <-- Comparison between multiple model inferences
│  │   │   │── 📂 predictions_base_model/
│  │   │   │── 📂 predictions_lora_new/
│  │   │── 📄 model_comparison.ipynb               <-- Metrics comparison visualized
│  │── 📄 Inference.py               <-- Run this to make an inference
│  │── 📄 lora_training.py           <-- Run this to fine-tune a model on the micro categories
│  │── 📄 Metrics_compute_2.py
│
📂 models/                             <-- Storage of the best models
│── 📂 lora_distill_llama_8b_boost/    <-- Micro category model
│── 📂 lora_distill_Qwen_1.5B_boost/   <-- Micro category model
│── 📂 Macro_lora_8B/                  <-- Macro category model
│
📂 src/config/                        <-- Put your Hugging Face API and Azure key here
│
📂 streamlit_app/               <-- Contains our streamlit app to vizualise the results 
│
.gitignore
│
Prompting_mistral_micro_classif.ipynb  <-- our prompting results 
│
first_classif.ipynb  <-- Classify between climate and non climate
│
Readme.md
│
requirements.txt
```

The code was tested using '''Python 3.9.13'''

## Getting Started

1. Put your API key in a config file you will create at `src/config/config.json`:

```json
{
    "huggingface": {
        "api_token": "XXX"
    },
    {

    "azure": {
        "AZURE_STORAGE_CONNECTION_STRING": "XXX",
    }

    },
}
```

2. Create a virtual environment:

```sh
python3 -m venv baroclimat
source baroclimat/bin/activate
pip install -r requirements.txt
```


## 🧠 News Classification Models

This repository contains several models trained for news classification into thematic categories.

## 📦 Available Models

- **`models/lora-distill-llama-8b-new`**  
  Trained on our newly labeled dataset:  
  `data/Annotations_sous_thematiques_new.csv`

- **`models/Macro-lora-8B`**  
  Trained on macro-categories derived from the new annotations:  
  `data/Annotations_macro_thematiques_new.csv`

- **`models/lora-distill-llama-8b-boost`**  
  Trained on the older annotation version:  
  `data/annotation_sous_thematiques.csv`

## 📰 Data

- **`data/2024_JT_TF1_F2.csv`**  
  Contains 2024 news bulletins that we classified and used in our Streamlit app.

---

## 🧪 Testing Your Trained Models

### ➤ For 25 Categories (Micro-thematic)

```bash
python3 Micro_category/Inference.py
```

### ➤ For 5 Categories (Macro-thematic)

```bash
python3 Macro_category/Inference_macro_1_cat.py
```

---

## 🏋️‍♂️ Training a Model

You can launch training with the following scripts:

```bash
python3 Micro_category/Inference.py
```

```bash
python3 Macro_category/Inference.py
```

---

## 🚀 Launching the Streamlit App

To visualize results or interact with the classifier via a UI:

```bash
streamlit run streamlit_app/app.py
```

