import os
from typing import Dict, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModel

from First_classif.utils.preprocessing import get_top_words_attention
from First_classif.utils.storage_connector import StorageConnector


class ClimateCamembert(torch.nn.Module):
    """Class for climate change topic detection based on Camembert pretrained model."""

    def __init__(self, tokenizer):
        super(ClimateCamembert, self).__init__()
        self.camembert = AutoModel.from_pretrained("camembert-base")
        self.dropout = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(768, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.tokenizer = tokenizer

    def forward(
        self, input_ids: torch.LongTensor, mask_ids: torch.LongTensor, token_type_ids: torch.LongTensor
    ) -> torch.LongTensor:
        """Forward pass of the model.

        Args:
                input_ids (torch.LongTensor): Shape (batch_size, sequence_length).
                        Indices of input sequence tokens in the vocabulary.
                mask_ids (torch.LongTensor): Shape (batch_size, sequence_length)).
                        Mask to avoid performing attention on padding token indices.
                token_type_ids (torch.LongTensor): Shape (batch_size, sequence_length)).
                        Segment token indices to indicate first and second portions of the inputs.
                        Indices are selected in [0, 1].

        Returns:
                torch.LongTensor : Shape (batch_size, 2) — Classification probabilities.
        """
        _, pooled_output, attention_layers = self.camembert(
            input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids, return_dict=False, output_attentions=True
        )
        x = self.dropout(pooled_output)
        x = self.dense(x)
        mean_attention = torch.mean(attention_layers[-1], axis=1)
        return x, mean_attention

    def from_pretrained(self, model_id: str, device: str, storage_connector: StorageConnector):
        """Load weights from a previously trained model."""
        path_model = os.path.join(storage_connector.container_name, model_id, f"{model_id}.pt")
        if os.path.exists(path_model):
            self.load_state_dict(torch.load(path_model, map_location=device))
            logger.info(f"Model loaded from {path_model}.")
        else:
            os.makedirs(os.path.dirname(path_model), exist_ok=True)
            logger.info("Model not found locally. Loading it from Azure Blob Storage.")
            storage_connector.download_file(os.path.join(model_id, f"{model_id}.pt"), path_model)
            self.load_state_dict(torch.load(path_model, map_location=device))

    def train_(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        device: str,
        weights: torch.LongTensor,
        path_dir_model_save: str,
    ):
        """Train the model.

        Args:
                train_loader (DataLoader): Training data loader.
                val_loader (DataLoader): Validation data loader.
                epochs (int): Number of epochs.
                lr (float): Learning rate.
                device (str): Device to train the model on.
                weights (torch.LongTensor): Class weights.
                path_dir_model_save (str): Path to the directory where to save the best model.
        """
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")
        best_loss = np.inf
        model_id = path_dir_model_save.split(os.sep)[-1]
        path_weight = os.path.join(path_dir_model_save, f"{model_id}.pt")
        for epoch in range(epochs):
            epoch_loss = 0
            all_good_pred = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, dtype=torch.long)
                mask_ids = batch["attention_mask"].to(device, dtype=torch.long)
                token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
                labels = batch["labels"].to(device, dtype=torch.long)

                self.zero_grad()

                logits, _ = self(input_ids, mask_ids, token_type_ids)
                preds = torch.argmax(logits, dim=1)

                loss = criterion(logits, labels.long())
                loss.backward()
                optimizer.step()
                good_pred = torch.sum(preds == labels)
                epoch_loss += loss.item()
                all_good_pred += good_pred.item()

            epoch_loss /= len(train_loader.dataset)
            all_good_pred /= len(train_loader.dataset)
            logger.info(f"Epoch {epoch + 1} loss is {epoch_loss:.4f} and accuracy is {all_good_pred:.4f}")
            loss_val, _, _, _ = self.evaluate(val_loader, device, weights)
            if loss_val < best_loss:
                best_loss = loss_val
                self.save(path_weight)

        logger.info(f"Training finished! Best weights saved to {path_weight}")

    def evaluate(
        self, dataloader: DataLoader, device: str, weights: torch.LongTensor
    ) -> Union[float, float, torch.LongTensor, torch.LongTensor]:
        """Evaluate the model.

        Args:
                dataloader (torch.utils.data.DataLoader): Data loader.
                device (str): Device to train the model on.
                weights (torch.LongTensor): Class weights.

        Returns:
                Union[float, float, torch.LongTensor, torch.LongTensor]: Validation loss, accuracy, preds, probas.
        """
        self.eval()
        self.to(device)
        weights = weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="mean")
        with torch.no_grad():
            loss_val = 0
            all_good_pred = 0
            all_preds = []
            all_preds_probas = []
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, dtype=torch.long)
                mask_ids = batch["attention_mask"].to(device, dtype=torch.long)
                token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
                labels = batch["labels"].to(device, dtype=torch.long)

                logits, _ = self(input_ids, mask_ids, token_type_ids)
                preds = torch.argmax(logits, dim=1)
                proba_prediction = torch.max(self.softmax(logits), axis=1).values
                proba_prediction = proba_prediction.detach().cpu()

                loss = criterion(logits, labels.long())
                good_pred = torch.sum(preds == labels)
                preds = preds.detach().cpu()
                loss_val += loss.item()
                all_good_pred += good_pred.item()
                all_preds.append(preds)
                all_preds_probas.append(proba_prediction)

            loss_val /= len(dataloader.dataset)
            accuracy = all_good_pred / len(dataloader.dataset)
            logger.info(f"Validation loss is {loss_val:.4f} and accuracy is {accuracy:.4f}")
        return loss_val, accuracy, torch.cat(all_preds), torch.cat(all_preds_probas)

    def predict(self, data: Union[DataLoader, Dict[str, torch.LongTensor]], device: str) -> tuple:
        """Predict on the data.

        Args:
                dataloader (Union[DataLoader, Dict[str, torch.LongTensor]]): Data loader or a single batch.
                device (str): Device to train the model on.

        Returns:
                np.ndarra: Shape (batch_size) — Predicted labels.
                np.ndarray: Shape (batch_size, 2) — Classification probabilities.
                list: List of top words attention for each document.
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                preds = []
                probas = []
                all_top_words_attention = []
                for batch in data:
                    input_ids = batch["input_ids"].to(device, dtype=torch.long)
                    mask_ids = batch["attention_mask"].to(device, dtype=torch.long)
                    token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)

                    logits_batch, mean_attention = self(input_ids, mask_ids, token_type_ids)
                    preds_batch = torch.argmax(logits_batch, dim=1)
                    preds_batch = preds_batch.detach().cpu().numpy()
                    proba_prediction = torch.max(self.softmax(logits_batch), axis=1).values
                    proba_prediction = proba_prediction.detach().cpu().numpy()
                    top_words_attention = get_top_words_attention(
                        mean_attention, batch["input_ids"], self.tokenizer, device
                    )

                    preds.extend(preds_batch)
                    probas.extend(proba_prediction)
                    all_top_words_attention.extend(top_words_attention)
            else:
                input_ids = data["input_ids"].to(device, dtype=torch.long)
                mask_ids = data["attention_mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

                logits, mean_attention = self(input_ids, mask_ids, token_type_ids)
                preds = torch.argmax(logits, dim=1)
                preds = preds.detach().cpu().numpy()
                probas = torch.max(self.softmax(logits), axis=1).values
                probas = probas.detach().cpu().numpy()
                all_top_words_attention = get_top_words_attention(
                    mean_attention, data["input_ids"], self.tokenizer, device
                )
        return preds, probas, all_top_words_attention

    def save(self, path: str):
        """Save the model.

        Args:
                path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
