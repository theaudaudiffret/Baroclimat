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
📂 Prompting_notebooks/               
📂 src/config/                        <-- Put your Hugging Face API key here
│
.gitignore
│
Readme.md
│
requirements.txt
```

## Getting Started

1. Put your API key in a config file you will create at `src/config/config.json`:

```json
{
    "huggingface": {
        "api_token": "XXX"
    }
}
```

2. Create a virtual environment:

```sh
python3 -m venv baroclima
source baroclima/bin/activate
pip install -r requirements.txt
```

## Testing Your Trained Models with 25 Categories

```sh
python3 Micro_category/Inference.py
```

## Training a Model on Macro Categories

Modify `lora_paths`, `save_paths`, `path_data`, then run:

```sh
python3 Macro_category/lora_training_MACRO.py
```

## To Do
- Verify paths
- Automate processes

