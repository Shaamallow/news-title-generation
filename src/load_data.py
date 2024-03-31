"""Load the data from the datasets"""

import re
import pandas as pd
from datasets import Dataset
from torch import Tensor

from src.metrics import Tokenizer

TRAIN_PATH = "./data/train.csv"
VALIDATION_PATH = "./data/validation.csv"
TEST_PATH = "./data/test_text.csv"


def load_data(
    train_path: str = TRAIN_PATH,
    validation_path: str = VALIDATION_PATH,
    test_path: str = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the data from the files
    ## Input :
    - train_path : str : The path to the training data
    - validation_path : str : The path to the validation data
    - test_path : str : The path to the test data
    ## Output :
    - train_df : pd.DataFrame : The training data
    - validation_df : pd.DataFrame : The validation data
    - test_df : pd.DataFrame : The test data
    """

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    return train_df, validation_df, test_df


def preprocess_text(
    text: str,
    title: str,
    tokenizer: Tokenizer,
    prefix: str = "summarize: ",
    max_input_length: int = 1024,
    max_target_length: int = 64,
) -> dict[str, Tensor]:
    """Preprocess the text data

    ## Input :
    - text : str : The text data to be preprocessed
    - title : str : The title of the text data, The Target
    - prefix : str : The prefix to be added to the text data as T5 model
        can be used for translation as well
    - max_input_length : int : The maximum length of the input text
    - max_target_length : int : The maximum length of the target text
    - tokenizer : t5_tokenizer : The tokenizer object

    ## Output :
    - model_inputs : Dict[str, Union[torch.Tensor, None]] : The model inputs
    """
    WHITESPACE_HANDLER = lambda k: re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))  # type: ignore

    inputs = tokenizer(
        WHITESPACE_HANDLER(f"{prefix} {text}"),
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    targets = tokenizer(
        WHITESPACE_HANDLER(title),
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )

    model_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": targets.input_ids,
    }

    return model_inputs


def preprocess_from_df(df: pd.DataFrame, tokenizer: Tokenizer):
    """Preprocess dataframe data using a tokenizer, for training purposes."""

    dataframe_list = []
    for i in range(len(df)):
        dataframe_list.append(
            preprocess_text(df["text"].iloc[i], df["titles"].iloc[i], tokenizer)
        )

    return Dataset.from_list(dataframe_list)
