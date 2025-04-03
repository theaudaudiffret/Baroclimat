import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from First_classif.config import BATCH_SIZE, LR, MAX_LEN, N_EPOCHS, STORAGE_CONTAINER_MODELS, TEST_SIZE
from First_classif.modeling.camembert_climate import ClimateCamembert
from First_classif.utils.preprocessing import prepare_train_dataset
from First_classif.utils.storage_connector import StorageConnector
from First_classif.utils.utils import save_param_run, save_perf_model

load_dotenv()


def train_model(
    path_dataset: str,
    test_size: float,
    max_len: int,
    learning_rate: float,
    n_epochs: int,
    batch_size: int,
    col_name_text: str,
    col_name_label: str,
    local_save: bool = False,
):
    """Train a model to classify to detect whether a text talks about climate change or not.

    Args:
        path_dataset (str): Path to the dataset to be used for training. (local file or sas url)
        col_name_text (str): Naming of the column of the csv where verbatims are stored.
        col_name_label (str): Naming of the column of the csv where intents are stored
        n_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        test_size (float): Share of dataset used for testing.
        max_len (int): Maximum length of the tokenized input sentence.
        local_save (bool, optional): Whether to also save the results locally in addition to the Azure Blob Storage.
                                    Defaults to False.
    """
    # Prepare dataset
    dataset = pd.read_csv(path_dataset)
    encoder = AutoTokenizer.from_pretrained("camembert-base")
    train_dataset, val_dataset, test_dataset = prepare_train_dataset(
        dataset, col_name_text, col_name_label, test_size, max_len, encoder
    )
    storage_connector = StorageConnector(os.getenv("AZURE_STORAGE_CONNECTION_STRING"), STORAGE_CONTAINER_MODELS)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(train_dataset.labels), y=train_dataset.labels),
        dtype=torch.float,
    )
    classifier = ClimateCamembert(tokenizer=encoder)
    time = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dir_save_path = os.path.join(STORAGE_CONTAINER_MODELS, time)
    os.makedirs(dir_save_path, exist_ok=True)
    save_param_run(
        dir_save_path,
        n_epochs,
        learning_rate,
        batch_size,
        weights.tolist(),
        max_len,
        train_dataset.data,
        val_dataset.data,
        test_dataset.data,
    )
    classifier.train_(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        epochs=n_epochs,
        lr=learning_rate,
        device=device,
        weights=weights,
        path_dir_model_save=dir_save_path,
    )

    # Evaluate model
    preds, _, _ = classifier.predict(
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0), device
    )
    confusion_mat = confusion_matrix(test_dataset.labels, preds)
    class_report = classification_report(test_dataset.labels, preds, zero_division="warn")

    # Save model and performance
    save_perf_model(confusion_mat, class_report, time, storage_connector, local_save)
    logger.success(
        f"Model trained and evaluated. Results saved in {dir_save_path} in Azure Blob Storage and locally if specified."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_dataset", type=str, help="Path to the dataset to be used for training. (csv file)")
    parser.add_argument(
        "--col_name_text", type=str, default="text", help="Naming of the column of the csv where verbatims are stored."
    )
    parser.add_argument(
        "--col_name_label", type=str, default="label", help="Naming of the column of the csv where intents are stored."
    )
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Share of dataset used for testing.")
    parser.add_argument("--max_len", type=int, default=MAX_LEN, help="Maximum length of the tokenized input sentence.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--local_save", action="store_true", help="Whether to also save the results locally.")
    args = parser.parse_args()
    train_model(
        path_dataset=args.path_dataset,
        col_name_text=args.col_name_text,
        col_name_label=args.col_name_label,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        test_size=args.test_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        local_save=args.local_save,
    )
