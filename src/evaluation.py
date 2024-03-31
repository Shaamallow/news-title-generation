"""Evaluate models and generate submissions"""

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from src.metrics import Tokenizer


def embeddings(text: pd.Series, toknizer: Tokenizer, batch_size: int = 16) -> pd.Series:
    """
    Generate embeddings using the given tokenizer and return the pd.Series of embeddings
    """
    embeddings = []

    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i : i + batch_size]
        input_text = toknizer(batch.tolist(), padding=True, truncation=True).input_ids
        embeddings.extend(input_text)

    return pd.Series(embeddings)


def summary(
    text: pd.Series,
    tokenizer: Tokenizer,
    model: AutoModelForSeq2SeqLM,
    batch_size: int = 64,
):
    """Generate summaries using the T5 model

    Using the standard representation defined in baseline part of notebook

    ## Input :
    - text : pd.Series : The text data for which the summaries are to be generated
        (the text column from the dataset)
    - t5_tokenizer : PreTrainedTokenizer : The tokenizer object
    - model : AutoModelForSeq2SeqLM : The model object

    ## Output :
    - summaries : List[Tuple[int, str]] : A list of Tuples containing the index of the text
        and the summary generated using the T5 model
    """
    summaries = []

    for i in tqdm(range(0, len(text), batch_size)):
        batch = text[i : i + batch_size]
        input_text = tokenizer(
            batch.tolist(), return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(
            model.device  # type: ignore
        )
        output = model.generate(  # type: ignore
            input_text,
            max_length=64,
            early_stopping=True,
            num_beams=4,
            num_return_sequences=1,
        )
        for idx, out in enumerate(output):
            summaries.append((i + idx, tokenizer.decode(out, skip_special_tokens=True)))
    return summaries
