import argparse
import datetime as dt
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from First_classif.config import (
    BATCH_SIZE,
    DEFAULT_EMBED_MODEL_ID,
    DEFAULT_MODEL_ID,
    MAX_LEN,
    STORAGE_CONTAINER_MODELS,
    STORAGE_CONTAINER_RESULTS,
)
from First_classif.dataset.text_dataset import CustomDataset
from First_classif.modeling.camembert_climate import ClimateCamembert
from First_classif.utils.storage_connector import StorageConnector
from First_classif.utils.utils import save_perf_model, save_results_inference

load_dotenv(override=True)


def inference(
    path_input_dataset: str,
    model_id: str,
    embed_model_id: str,
    max_len: int,
    col_name_text: str,
    col_name_label: str,
    batch_size: int,
    evaluate_classifier: bool = False,
    local_save: bool = False,
):
    """States whether a text talks about climate change or not.
      On top of this, perform a semi-supervised method to classify in finer categories of climate related topics.

    Args:
        path_input_dataset (str): Path to the input file containing the sentences to classify. (local file or sas url)
        model_id (str): Id of the model to use for the climate change detection.
        embed_model_id (str): Id of the fasttext model to use for the climate topics classification.
        col_name_text (str, optional):  Naming of the column of the csv where verbatims are stored.
        col_name_label (str, optional): Naming of the column of the csv where intents are stored.
        max_len (int, optional): Maximum length of the tokenized input sentence you used for training.
        batch_size (int, optional): Batch size for the inference task.
        evaluate_classifier (bool, optional): Whether to evaluate the classifier on your input.
                                                If True, the input file must contain a column "label".
                                                Defaults to False.
        local_save (bool, optional): Whether to also save the results locally in addition to the Azure Blob Storage.
                                    Defaults to False.
    """
    # Load models and input sentences
    input_dataset = pd.read_csv(path_input_dataset)
    if input_dataset[col_name_text].isnull().sum() > 0:
        logger.warning(f"Found {input_dataset[col_name_text].isnull().sum()} empty text rows. Removing them.")
        input_dataset = input_dataset.dropna(subset=[col_name_text])

    logger.info("Loading pretrained model and embedding model, this may take a while.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    storage_connector = StorageConnector(os.getenv("AZURE_STORAGE_CONNECTION_STRING"), STORAGE_CONTAINER_MODELS)

    encoder = AutoTokenizer.from_pretrained("camembert-base")
    classifier = ClimateCamembert(tokenizer=encoder)
    classifier.from_pretrained(model_id, device, storage_connector)

    # Process input text
    if not evaluate_classifier:
        col_name_label = None
    inference_dataset = CustomDataset(input_dataset, encoder, max_len, col_name_label, col_name_text)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Perform inference
    time = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger.info("Performing first step of inference: Climate change detection.")
    y_pred, y_pred_probas, top_words = classifier.predict(inference_loader, device)
    input_dataset["climate_related"] = y_pred
    input_dataset["confidence"] = y_pred_probas
    input_dataset["top_words"] = top_words

    # Save results
    storage_connector.container_name = STORAGE_CONTAINER_RESULTS
    if evaluate_classifier:
        class_report = classification_report(input_dataset[col_name_label], y_pred, zero_division="warn")
        conf_matrix = confusion_matrix(input_dataset[col_name_label], y_pred)
        save_perf_model(conf_matrix, class_report, id=time, storage_connector=storage_connector, local_save=local_save)

    save_results_inference(input_dataset, time, storage_connector, local_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_input_dataset", type=str, help="Path to the input file containing the text to classify. (csv file)"
    )
    parser.add_argument(
        "--model_id", type=str, help="Id of the model to use for the inference.", default=DEFAULT_MODEL_ID
    )
    parser.add_argument(
        "--embed_model_id",
        type=str,
        help="Id of the fasttext model to use for the climate topics classification.",
        default=DEFAULT_EMBED_MODEL_ID,
    )
    parser.add_argument(
        "--col_name_text", type=str, default="text", help="Naming of the column of the csv where verbatims are stored."
    )
    parser.add_argument(
        "--col_name_label", type=str, default="label", help="Naming of the column of the csv where intents are stored."
    )
    parser.add_argument(
        "--evaluate_classifier",
        action="store_true",
        help="Whether to evaluate the classifier on your input. If True, the input file must contain a column 'label'.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=MAX_LEN,
        help="Maximum length of the tokenized input sentence you used for training. Default to 512.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for the inference task. Defaults to 16."
    )
    parser.add_argument("--local_save", action="store_true", help="Whether to also save the results locally.")

    args = parser.parse_args()

    inference(
        path_input_dataset=args.path_input_dataset,
        model_id=args.model_id,
        embed_model_id=args.embed_model_id,
        max_len=args.max_len,
        batch_size=args.batch_size,
        col_name_text=args.col_name_text,
        col_name_label=args.col_name_label,
        evaluate_classifier=args.evaluate_classifier,
        local_save=args.local_save,
    )
