import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from First_classif.config import LABELS_CLIMATE
from First_classif.utils.storage_connector import StorageConnector


def save_perf_model(
    confusion_matrix: np.ndarray,
    class_report: str,
    id: str,
    storage_connector: StorageConnector,
    local_save: bool = False,
):
    """Save the performance of the model in a txt file.

    Args:
        confusion_matrix (np.ndarray): Confusion matrix of the model.
        class_report (str): Classification report of the model.
        id (str): Id of the run.
        storage_connector (StorageConnector): Storage connector to upload the results to Azure Blob Storage.
        local_save (bool, optional): Whether to also save the results locally in addition to the Azure Blob Storage.
                                    Defaults to False.
    """
    dirpath_results = os.path.join("data", storage_connector.container_name, id)
    os.makedirs(dirpath_results, exist_ok=True)
    with open(os.path.join(dirpath_results, "performance.txt"), "w") as f:
        f.write("Classification report:\n")
        f.write(class_report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS_CLIMATE, yticklabels=LABELS_CLIMATE
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(dirpath_results, "confusion_matrix.png"))
    storage_connector.upload_directory(dirpath_results, id)
    if not local_save:
        shutil.rmtree(dirpath_results, ignore_errors=True)


def save_results_inference(
    inference_data: pd.DataFrame, inference_id: str, storage_connector: StorageConnector, local_save: bool = False
):
    """Save the results of the inference in a csv file.

    Args:
        inference_data (pd.DataFrame): Dataframe containing the results of the inference.
        inference_id (str): Id of the inference.
        storage_connector (StorageConnector): Storage connector to upload the results to Azure Blob Storage.
        local_save (bool, optional): Whether to also save the results locally in addition to the Azure Blob Storage.
                                            Defaults to False.
    """
    output = inference_data.to_csv(encoding="utf-8")
    blob_client = storage_connector.container_client.get_blob_client(f"{inference_id}/inference.csv")
    blob_client.upload_blob(output, overwrite=True)
    if local_save:
        dir_path_results = os.path.join("data", storage_connector.container_name, inference_id)
        os.makedirs(dir_path_results, exist_ok=True)
        inference_data.to_csv(os.path.join(dir_path_results, "inference.csv"), index=False)
        logger.success(f"Inference results saved in {dir_path_results} locally and in Azure Blob Storage.")
    else:
        logger.success(
            f"""Inference results saved in {storage_connector.container_name}/{inference_id}/inference.csv
            in Azure Blob Storage."""
        )


def save_param_run(
    dirpath_results: str,
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    weights: list,
    max_len: int,
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
):
    """Save the parameters of the run in a txt file.

    Args:
        dirpath_results (str): Path to the directory where you want to store the results of the inference.
        n_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        weights (list): Weights of the classes.
        max_len (int): Maximum length of the sequences.
        train_dataset (pd.DataFrame): Training dataset.
        val_dataset (pd.DataFrame): Validation dataset.
        test_dataset (pd.DataFrame): Test dataset.
    """
    train_dataset.to_csv(os.path.join("data", dirpath_results, "train_dataset.csv"), index=False)
    val_dataset.to_csv(os.path.join("data", dirpath_results, "val_dataset.csv"), index=False)
    test_dataset.to_csv(os.path.join("data", dirpath_results, "test_dataset.csv"), index=False)
    params_run = {
        "lr": learning_rate,
        "batch_size": batch_size,
        "epoch": n_epochs,
        "max_len": max_len,
        "weights": weights,
        "len_dataset": len(train_dataset) + len(val_dataset) + len(test_dataset),
    }
    with open(os.path.join("data", dirpath_results, "params.json"), "w") as f:
        json.dump(params_run, f)
