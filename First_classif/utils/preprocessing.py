from typing import Tuple

import pandas as pd
import torch
from deep_translator import GoogleTranslator
from loguru import logger
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from unidecode import unidecode

from First_classif.config import ENGLISH_STOPWORDS, FRENCH_STOPWORDS
from First_classif.dataset.text_dataset import CustomDataset


def translate_text_to_english(text: str) -> str:
    """
    This function translates a text from French to English. It will be used in the following to feed BERT
    (climateBERT) with English texts

    Input:
    - text (string): French text to be translated

    Output:
    - translated_text (string): translated to English text
    """
    lang_target = "en"
    lang_source = "fr"

    max_length = 4999  # Maximal number of characters in the text for Google Translate to handle it
    translator = GoogleTranslator(source=lang_source, target=lang_target)

    # Handle long texts
    if len(text) >= max_length:
        # Divide the text in chunks of admissible sizes if it is too long
        translated_text = ""
        nb_chunks = len(text) // max_length + 1

        # Translate each chunk and concatenate the translations
        for i in range(nb_chunks):
            chunk = text[i * max_length : (i + 1) * max_length]
            translated_chunk = translator.translate(chunk)
            translated_text += translated_chunk
            translated_text += " "

    else:
        translated_text = translator.translate(text)

    return translated_text


def batch_translation(
    content_df: pd.DataFrame, col_text_source: str = "text", col_text_out: str = "text_en", length_batch: int = 10
) -> pd.DataFrame:
    """
    This function translates French texts to English in a DataFrame.
    The translation is made by batch to successfully handle large DataFrames

    Inputs:
    - content_df (DataFrame): DataFrame containing the texts to be translated
    - col_text_source (string): name of the DataFrame column with the text to be translated
    - col_text_out (string): name of the column to create in the DataFrame to store the translated texts
    - length_batch (int): length of the translation batches

    Output:
    - content_df (DataFrame): input DataFrame with an additional column col_text_out containing the translated texts
    """
    nb_batch = len(content_df) // length_batch

    translated_text = pd.DataFrame()

    # Translate texts batchwise
    for i in tqdm(range(nb_batch)):
        if len(content_df) > length_batch:
            current_df = content_df.iloc[i * length_batch : (i + 1) * length_batch, :]
        else:
            current_df = content_df

        translated_batch = current_df[col_text_source].apply(translate_text_to_english)
        translated_text = pd.concat([translated_text, translated_batch])

    translated_text.columns = [col_text_out]
    content_df[col_text_out] = translated_text[col_text_out]

    return content_df


def prepare_train_dataset(
    dataset: pd.DataFrame, col_name_text: str, col_name_label: str, test_size: float, max_len: int, encoder
) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
    """Prepare the dataset for training by splitting it into training, validation, and test sets.

    Args:
        dataset (pd.DataFrame): Dataset to be used for training.
        col_name_text (str): Name of the column containing the text data.
        col_name_label (str): Name of the column containing the label data.
        test_size (float): Share of dataset used for testing.
        max_len (int): Maximum length of the tokenized input sentence.
        encoder: Tokenizer to be used for encoding the text data.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    if dataset[col_name_text].isnull().sum() > 0:
        logger.warning(f"Found {dataset[col_name_text].isnull().sum()} empty text rows. Removing them.")
        dataset = dataset.dropna(subset=[col_name_text])
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        dataset.loc[:, dataset.columns != col_name_label],
        dataset.loc[:, [col_name_label]],
        random_state=42,
        test_size=test_size,
        stratify=dataset[col_name_label],
    )

    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=42, test_size=0.5, stratify=temp_labels
    )
    train_dataset = CustomDataset(
        pd.concat([train_text, train_labels], axis=1).reset_index(drop=True),
        encoder,
        max_len,
        col_name_label,
        col_name_text,
    )
    val_dataset = CustomDataset(
        pd.concat([val_text, val_labels], axis=1).reset_index(drop=True),
        encoder,
        max_len,
        col_name_label,
        col_name_text,
    )
    test_dataset = CustomDataset(
        pd.concat([test_text, test_labels], axis=1).reset_index(drop=True),
        encoder,
        max_len,
        col_name_label,
        col_name_text,
    )
    return train_dataset, val_dataset, test_dataset


def format_document(document: str, method_used: str = "lemmatization", langage: str = "french") -> list[str]:
    """
    Tokenize the input document and perform the following steps:
    1. Convert the document to lowercase and remove accents.
    2. Remove French stop words.
    3. Remove special characters and punctuation.
    4. Lemmatize the words.

    Args:
    - document (str): The input document to be tokenized.
    - method_used (str): The method used for tokenization. Choose between 'lemmatization' and 'stemming'.
    - langage (str): The language used in the document. Choose between 'french' and 'english'.

    Returns:
    - list[str]: A list of tokens based on the chosen method.
    """
    # Lower case all the words
    document = unidecode(document.lower())

    # Tokenization
    document = word_tokenize(document)

    # Stop Word Removal
    if langage == "french":
        stop_words = FRENCH_STOPWORDS
        stop_words = set(
            [unidecode(word) for word in stop_words]
            + ["apres", "comme", "selon", "nbsp", "jamais", "journal", "sujets", "theme", "televise"]
        )
    elif langage == "english":
        stop_words = ENGLISH_STOPWORDS
        stop_words = set(
            [unidecode(word) for word in stop_words]
            + ["after", "like", "according to", "nbsp", "never", "journal", "subjects", "topic", "television"]
        )
    else:
        raise ValueError("Language not implemented yet. Please choose between 'french' and 'english'.")
    document = [word for word in document if word not in stop_words]

    # Removing Special Characters and Punctuation
    document = [word for word in document if word.isalpha()]

    # Lemmatization
    if method_used == "lemmatization":
        lemm = WordNetLemmatizer()
        document = [lemm.lemmatize(word) for word in document]
    elif method_used == "stemming":
        stem = PorterStemmer()
        document = [stem.lemmatize(word) for word in document]
    else:
        raise ValueError("Method not implemented yet. Please choose between 'lemmatization' and 'stemming'.")

    return document


def get_top_words_attention(
    mean_attention: torch.LongTensor, input_ids: torch.LongTensor, tokenizer, device: str, nb_words: int = 3
) -> list[list[str]]:
    """Get the top words from the attention matrix for a batch of sentences.

    Args:
        mean_attention (torch.LongTensor): Mean attention matrix of the last layer of the model.
        input_ids (torch.LongTensor): Input ids of the batch of sentences.
        tokenizer (_type_): Tokenizer used to convert ids to words.
        device (str): Device to be used for the computations.
        nb_words (int, optional): Number of top words to keep. Defaults to 3:int.

    Returns:
        list[list[str]]: List of top words for each sentence in the batch.
    """
    mean_attention = mean_attention.to(device)
    input_ids = input_ids.to(device)
    top_index_tokens = torch.topk(mean_attention[:, 0, :], k=100, dim=1).indices
    top_words_ids = input_ids[torch.arange(top_index_tokens.size(0)).unsqueeze(1), top_index_tokens]
    top_words_text = [
        tokenizer.convert_ids_to_tokens(top_words_ids[batch_index, :]) for batch_index in range(len(top_words_ids))
    ]
    stop_words = list(FRENCH_STOPWORDS) + ["</s>", "<s>"]
    top_words_text = [[word.replace("▁", "") for word in words] for words in top_words_text]
    top_words_text = [
        [word for word in words if (word not in stop_words) and word.isalpha()] for words in top_words_text
    ]
    top_words_text = [words[:nb_words] for words in top_words_text]
    return top_words_text
