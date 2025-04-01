Put your API key in a config file you will create at src/config/config.json: 
{
    "huggingface": {
        "api_token": "XXX"
    }
}


To test your trained models with 25 categories: 

python3 Micro_category/Inference.py


Indicate in lora_paths, save_paths, path_data 


python3 Macro_category/lora_training_MACRO.py


To do: verify paths, automation 