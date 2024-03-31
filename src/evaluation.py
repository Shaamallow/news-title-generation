"""Evaluate models and generate submissions"""

from transformers import AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
from src.metrics import Tokenizer


def t5_summary(
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
            num_return_sequences=1,
        )
        for idx, out in enumerate(output):
            summaries.append((i + idx, tokenizer.decode(out, skip_special_tokens=True)))
    return summaries
