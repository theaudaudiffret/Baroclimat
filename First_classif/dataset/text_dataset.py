from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Dataset class for training and evaluating transformer based models with pytorch library."""

    def __init__(self, data: pd.DataFrame, encoder, max_len: int, col_name_label: Optional[str], col_name_text: str):
        """
        Args:
            data (pd.DataFrame): Dataframe with columns "text" and "label"
            encoder (_type_): Pretrained tokenizer from transformers library
            max_len (int): Maximum length of the tokenized input sentence
            col_name_label (Optional[str]): Name of the column containing the labels. If None, no labels are used
            col_name_text (str): Name of the column containing the text

        """
        self.encoder = encoder
        self.data = data
        self.text = self.data[col_name_text].tolist()
        self.labels = self.data[col_name_label].tolist() if col_name_label else None
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.text[index]

        inputs = self.encoder(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        mask_ids = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        encoded_item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        if self.labels:
            encoded_item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return encoded_item
