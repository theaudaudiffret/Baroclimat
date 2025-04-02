Repository organization:


📂 data/                            <-- Labeled data used for training and computing metrics
📂 Macro_category/                  <-- sub category segmentation using 5 macro categories
│  │── 📂 Inference/                <-- csv containing the results of the inference: text + categories
│  │── 📂 Metrics/                  <-- csv containing the metrics for the model
│  │── 📄 Inference.py              <-- run this to make an inference
│  │── 📄 lora_training_MACRO.py    <-- run this to fine tune a model on the macro categories
│
📂 Micro_category/                  <-- sub category segmentation using 25 macro categories
│  │── 📂 Inference/                <-- csv containing the results of the inference: text + categories
│  │── 📂 Metrics/                  <-- csv containing the metrics for the model
│  │── 📂 Models_comparison/        <-- Comparison of fine tuning different models
│  │   │── 📂 Final_model/
│  │   │   │── 📂 inference_final_model/          <-- Inference of the 2 best models 
│  │   │   │── 📂 lora_distill_llama_8b_boost2/   <-- Inference of the 2 best models 
│  │   │   │── 📂 lora_distill_Qwen_1.5B_boost/
│  │   │   │── 📄 Benchmark.csv
│  │   │   │── 📄 resultats_base.csv
│  │   │   │── 📄 resultats_lora.csv
│  │   │── 📂 original_annotation_dataset/         <-- Comparison between multiple models inference
│  │   │   │── 📂 predictions_base_model/
│  │   │   │── 📂 predictions_lora_new/
│  │   │── 📄 model_comparison.ipynb               <-- Metrics comparison vizualised 
│  │── 📄 Inference.py               <-- run this to make an inference
│  │── 📄 lora_training.py           <-- run this to fine tune a model on the micro categories
│  │── 📄 Metrics_compute_2.py
│
📂 models/                             <--Storage of the best models
│── 📂 lora_distill_llama_8b_boost/    <--Micro category model 
│── 📂 lora_distill_Qwen_1.5B_boost/   <--Micro category model 
│── 📂 Macro_lora_8B/                  <--Macro category model 
│
📂 Prompting_notebooks/               
📂 src/config/                        <--Put your hugging face api key here 
│
.gitignore
│
Readme.md
│
requirements.txt





To get started: 




Put your API key in a config file you will create at src/config/config.json: 
{
    "huggingface": {
        "api_token": "XXX"
    }
}


Create a virtual environment: 

python3 -m venv baroclima
source baroclima/bin/activate

pip install -r requirements.txt






To test your trained models with 25 categories: 

python3 Micro_category/Inference.py


Indicate in lora_paths, save_paths, path_data 


python3 Macro_category/lora_training_MACRO.py


To do: verify paths, automation 